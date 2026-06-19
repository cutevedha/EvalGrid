"""
evalgrid/cache.py: On-disk score cache so repeated evals do not re-call the LLM.

Why caching matters
-------------------
LLM-based judges and reference graders cost real money and add latency. A score
is fully determined by (metric_name, test_case input/expected/context, output).
The cache hashes that tuple and skips the score computation when it's a hit.

Usage
-----
    from evalgrid import ScoreCache

    cache = ScoreCache(".evalgrid_cache")
    evaluate(cases, metrics=["correctness"], cache=cache)

    cache.stats()   # → {"hits": 12, "misses": 3, "saved_calls": 12}

Disable entirely by passing ``cache=False`` to ``evaluate``.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ScoreCache:
    """File-backed cache for metric scores. Thread-safe across processes via flock-less write-then-rename."""

    def __init__(self, directory: str = ".evalgrid_cache") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stats: Dict[str, int] = {"hits": 0, "misses": 0, "writes": 0}

    def get(self, metric_name: str, test_case: Any, actual_output: str) -> Optional[Dict[str, float]]:
        """Return cached scores for (metric, case, output) tuple, or None on miss."""
        key = self._make_key(metric_name, test_case, actual_output)
        path = self.directory / f"{key}.json"
        if not path.exists():
            self._stats["misses"] += 1
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            self._stats["hits"] += 1
            return entry.get("scores")
        except (json.JSONDecodeError, OSError):
            self._stats["misses"] += 1
            return None

    def put(self, metric_name: str, test_case: Any, actual_output: str, scores: Dict[str, float]) -> None:
        """Store scores in the cache."""
        key = self._make_key(metric_name, test_case, actual_output)
        path = self.directory / f"{key}.json"
        entry = {
            "metric": metric_name,
            "scores": scores,
            "timestamp": time.time(),
        }
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(entry, f)
            tmp.replace(path)
            self._stats["writes"] += 1
        except OSError:
            pass

    def stats(self) -> Dict[str, int]:
        """Return cache statistics for the current session."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "total": total,
            "hit_rate": round(self._stats["hits"] / total, 4) if total else 0.0,
        }

    def clear(self) -> int:
        """Delete every cached entry. Returns the number of files removed."""
        count = 0
        for path in self.directory.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        return count

    def _make_key(self, metric_name: str, test_case: Any, actual_output: str) -> str:
        case_input    = getattr(test_case, "input", "") or ""
        case_expected = getattr(test_case, "expected_output", None) or ""
        case_context  = getattr(test_case, "context", None) or ""
        payload = "".join([metric_name, case_input, case_expected, case_context, actual_output])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
