# Change Management - Category 9
# Any change to prompts, rubrics, metrics, or judge models requires revalidation; baselines
# are preserved across versions; breaking changes are flagged; and the pipeline can be
# rolled back. Built on content-hash versioning so a change is detected, not declared.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from governance.audit import content_hash


# ============================================================================
# COMPONENT VERSIONS  (what a run depended on)
# ============================================================================

@dataclass
class ComponentVersions:
    """Content-hash versions of every component whose change should force revalidation."""
    prompts: str = ""
    rubrics: str = ""
    metrics: str = ""
    judge: str = ""

    @classmethod
    def of(cls, prompts: Any = "", rubrics: Any = "", metrics: Any = "", judge: Any = "") -> "ComponentVersions":
        return cls(
            prompts=content_hash(prompts),
            rubrics=content_hash(rubrics),
            metrics=content_hash(metrics),
            judge=content_hash(judge),
        )

    def diff(self, other: "ComponentVersions") -> List[str]:
        """Return the names of components that changed between two version sets."""
        changed = []
        for field_name in ("prompts", "rubrics", "metrics", "judge"):
            if getattr(self, field_name) != getattr(other, field_name):
                changed.append(field_name)
        return changed


def requires_revalidation(previous: ComponentVersions, current: ComponentVersions) -> List[str]:
    """List the changed components that mandate re-running validation (empty = no change)."""
    return previous.diff(current)


# ============================================================================
# BASELINE STORE + ROLLBACK
# ============================================================================

@dataclass
class Baseline:
    version: str
    versions: ComponentVersions
    summary: Dict[str, Any]              # e.g. {"pass_rate": 0.91, ...}
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


class BaselineStore:
    """
    Preserves baselines across framework versions so comparisons survive upgrades, and
    supports rollback to any prior baseline.
    """

    def __init__(self):
        self._baselines: Dict[str, Baseline] = {}
        self._order: List[str] = []

    def save(self, version: str, versions: ComponentVersions, summary: Dict[str, Any]) -> None:
        self._baselines[version] = Baseline(version=version, versions=versions, summary=summary)
        if version not in self._order:
            self._order.append(version)

    def get(self, version: str) -> Optional[Baseline]:
        return self._baselines.get(version)

    def latest(self) -> Optional[Baseline]:
        return self._baselines[self._order[-1]] if self._order else None

    def rollback_to(self, version: str) -> Baseline:
        """Return a prior baseline as the active one (full-pipeline rollback target)."""
        if version not in self._baselines:
            raise KeyError(f"no baseline for version {version}")
        # Truncate history after the rollback target.
        idx = self._order.index(version)
        self._order = self._order[: idx + 1]
        return self._baselines[version]

    def compare(self, version_a: str, version_b: str, metric: str = "pass_rate") -> Dict[str, Any]:
        """Compare a summary metric between two baselines and flag regressions."""
        a, b = self.get(version_a), self.get(version_b)
        if not a or not b:
            return {"error": "missing baseline"}
        va, vb = a.summary.get(metric, 0.0), b.summary.get(metric, 0.0)
        return {
            "metric": metric,
            version_a: va,
            version_b: vb,
            "delta": round(vb - va, 4),
            "regression": vb < va,
        }


# ============================================================================
# BREAKING CHANGES
# ============================================================================

@dataclass
class BreakingChange:
    component: str
    description: str
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


def mark_breaking_changes(changed_components: List[str], descriptions: Optional[Dict[str, str]] = None) -> List[BreakingChange]:
    """Turn a list of changed components into breaking-change records for the report."""
    descriptions = descriptions or {}
    return [BreakingChange(component=c, description=descriptions.get(c, f"{c} changed")) for c in changed_components]
