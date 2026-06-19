# Data Integrity - Category 2
# Dataset version IDs + immutable snapshots, train/val/test leakage checks, duplicate /
# near-duplicate / contamination detection, and per-sample provenance. Operates on the
# framework's existing list-of-dict test cases (as produced by synthetic/dataset_builder),
# so it complements DatasetBuilder rather than replacing it.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from governance.audit import content_hash


# ============================================================================
# NORMALISATION
# ============================================================================

def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace: the canonical form for overlap comparisons."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> Set[str]:
    return set(_normalize(text).split())


def _jaccard(a: str, b: str) -> float:
    tokens_a, tokens_b = _tokens(a), _tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# ============================================================================
# IMMUTABLE DATASET SNAPSHOT  (versioning)
# ============================================================================

@dataclass
class DatasetSnapshot:
    """
    An immutable, content-addressed snapshot of an evaluation dataset.

    The ``version_id`` is a hash of the (name + samples), so any change to the data
    necessarily changes the version: you cannot silently alter a dataset and keep its ID.
    ``parent_version`` records lineage across revisions.
    """
    name: str
    samples: List[Dict[str, Any]]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    parent_version: Optional[str] = None
    provenance: Dict[str, str] = field(default_factory=dict)  # sample_id -> source
    version_id: str = ""

    def __post_init__(self):
        if not self.version_id:
            self.version_id = self._compute_version()

    def _compute_version(self) -> str:
        return content_hash({"name": self.name, "samples": self.samples})

    def verify(self) -> bool:
        """True if the data still matches its version ID (detects out-of-band tampering)."""
        return self.version_id == self._compute_version()

    def record_provenance(self, sample_id: str, source: str) -> None:
        """Attach an origin to a sample (e.g. 'redteam:jailbreak', 'human:annotator_3')."""
        self.provenance[sample_id] = source

    def save(self, path: str) -> str:
        """Persist the snapshot to JSON; returns the version ID."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "name": self.name,
                "version_id": self.version_id,
                "created_at": self.created_at,
                "parent_version": self.parent_version,
                "provenance": self.provenance,
                "samples": self.samples,
            }, f, indent=2, ensure_ascii=False)
        return self.version_id

    @classmethod
    def load(cls, path: str) -> "DatasetSnapshot":
        """Load a snapshot and verify its integrity on read."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        snap = cls(
            name=data["name"],
            samples=data["samples"],
            created_at=data.get("created_at", ""),
            parent_version=data.get("parent_version"),
            provenance=data.get("provenance", {}),
            version_id=data.get("version_id", ""),
        )
        if not snap.verify():
            raise ValueError(f"Snapshot integrity check failed for {path}: data does not match version_id")
        return snap

    def revise(self, samples: List[Dict[str, Any]]) -> "DatasetSnapshot":
        """Create a child snapshot from new samples, linked to this one's version."""
        return DatasetSnapshot(name=self.name, samples=samples, parent_version=self.version_id)


# ============================================================================
# LEAKAGE / DUPLICATE / CONTAMINATION CHECKS
# ============================================================================

def _input_of(sample: Dict[str, Any]) -> str:
    """Return the input text for a sample, or empty string when absent."""
    return sample.get("input", "")


def detect_split_leakage(splits: Dict[str, List[Dict[str, Any]]], near_threshold: float = 0.95) -> Dict[str, List[str]]:
    """
    Detect train/val/test leakage: samples that appear (exactly or near-exactly) in more
    than one split.

    Args:
        splits: mapping of split name -> samples (e.g. {"train": [...], "test": [...]}).
        near_threshold: Jaccard ≥ this counts as the "same" sample across splits.

    Returns:
        mapping "split_a|split_b" -> list of human-readable overlap descriptions.
    """
    names = list(splits)
    leaks: Dict[str, List[str]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair_leaks = []
            norms_b = {sid: _normalize(_input_of(s)) for sid, s in _ids(splits[b])}
            for sid_a, sa in _ids(splits[a]):
                na = _normalize(_input_of(sa))
                for sid_b, nb in norms_b.items():
                    if na == nb or _jaccard(na, nb) >= near_threshold:
                        pair_leaks.append(f"{a}:{sid_a} ↔ {b}:{sid_b}")
                        break
            if pair_leaks:
                leaks[f"{a}|{b}"] = pair_leaks
    return leaks


def find_duplicates(samples: List[Dict[str, Any]]) -> List[List[str]]:
    """Group sample IDs whose inputs are exact (normalised) duplicates."""
    buckets: Dict[str, List[str]] = {}
    for sid, s in _ids(samples):
        key = content_hash(_normalize(_input_of(s)))
        buckets.setdefault(key, []).append(sid)
    return [ids for ids in buckets.values() if len(ids) > 1]


def find_near_duplicates(samples: List[Dict[str, Any]], threshold: float = 0.85) -> List[Tuple[str, str, float]]:
    """Return (id_a, id_b, similarity) for distinct sample pairs above the similarity threshold."""
    indexed = _ids(samples)
    pairs = []
    for i in range(len(indexed)):
        sid_a, sa = indexed[i]
        for j in range(i + 1, len(indexed)):
            sid_b, sb = indexed[j]
            sim = _jaccard(_input_of(sa), _input_of(sb))
            if 1.0 > sim >= threshold:  # exclude exact dupes (handled separately)
                pairs.append((sid_a, sid_b, round(sim, 3)))
    return pairs


def detect_contamination(samples: List[Dict[str, Any]], reference_corpus: List[str],
                         threshold: float = 0.9) -> List[str]:
    """
    Flag samples whose input closely matches any text in a reference corpus (e.g. the
    model's known training data or canary strings): i.e. likely contaminated.
    """
    flagged = []
    for sid, s in _ids(samples):
        text = _input_of(s)
        for ref in reference_corpus:
            if _jaccard(text, ref) >= threshold:
                flagged.append(sid)
                break
    return flagged


def integrity_report(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One-call summary of duplicates and near-duplicates for a dataset."""
    dups = find_duplicates(samples)
    near = find_near_duplicates(samples)
    return {
        "total_samples": len(samples),
        "exact_duplicate_groups": dups,
        "exact_duplicate_count": sum(len(g) - 1 for g in dups),
        "near_duplicate_pairs": near,
        "clean": not dups and not near,
    }


# ============================================================================
# HELPERS
# ============================================================================

def _ids(samples: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Pair each sample with its string id, generating a stable fallback id when missing."""
    return [(str(s.get("id", f"idx{idx}")), s) for idx, s in enumerate(samples)]
