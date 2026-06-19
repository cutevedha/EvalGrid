# Scoring Guardrails - Category 4
# Deterministic scoring, defined behaviour for missing/malformed/truncated outputs,
# documented aggregation (weights + exclusions), tie-breaking, and outlier handling that
# is fixed *before* results are reviewed. None of this changes the target's output: it
# only governs how an already-produced output is turned into a number.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


# ============================================================================
# OUTPUT CONDITION HANDLING
# ============================================================================

# Recognised output conditions and how the policy treats them.
OUTPUT_OK = "ok"
OUTPUT_MISSING = "missing"
OUTPUT_MALFORMED = "malformed"
OUTPUT_TRUNCATED = "truncated"


def classify_output(
    output: Optional[str],
    max_len: Optional[int] = None,
    must_be_json: bool = False,
) -> str:
    """
    Classify an output before scoring so degenerate cases get explicit, consistent handling.

    Returns one of OUTPUT_OK / MISSING / MALFORMED / TRUNCATED.
    """
    if output is None or output == "":
        return OUTPUT_MISSING
    if output.startswith("Error:"):           # adapters surface failures as "Error: ..."
        return OUTPUT_MALFORMED
    if must_be_json:
        import json
        try:
            json.loads(output)
        except Exception:
            return OUTPUT_MALFORMED
    if max_len is not None and len(output) >= max_len:
        return OUTPUT_TRUNCATED
    return OUTPUT_OK


@dataclass
class ScoringPolicy:
    """
    A pre-declared, deterministic scoring policy for a suite.

    Every degenerate condition maps to an explicit score so nothing is left to chance, and
    aggregation/outlier rules are fixed up front (before anyone sees results).
    """
    missing_score: float = 0.0           # Score for absent output
    malformed_score: float = 0.0         # Score for unparseable/errored output
    truncated_penalty: float = 0.5       # Multiplier applied to truncated outputs
    tie_breaker: str = "lower"           # On ties prefer "lower" (conservative) or "higher"
    weights: Dict[str, float] = field(default_factory=dict)   # Per-metric aggregation weights
    exclusions: List[str] = field(default_factory=list)       # Metrics excluded from aggregation
    outlier_method: str = "none"         # "none" | "iqr": fixed before review

    def score_for_condition(self, condition: str, raw_score: float) -> float:
        """Map a raw score through the policy given the output condition."""
        if condition == OUTPUT_MISSING:
            return self.missing_score
        if condition == OUTPUT_MALFORMED:
            return self.malformed_score
        if condition == OUTPUT_TRUNCATED:
            return raw_score * self.truncated_penalty
        return raw_score


# ============================================================================
# DETERMINISM GUARANTEE
# ============================================================================

def assert_deterministic(scorer: Callable[[], float], runs: int = 3) -> bool:
    """
    Verify a scorer returns identical results for identical inputs across repeated calls.

    Returns True if every run agrees: the spec's "scoring rules are deterministic for
    identical inputs". Used in regression tests and CI.
    """
    first = scorer()
    return all(scorer() == first for _ in range(runs - 1))


def break_tie(candidates: Sequence[float], rule: str = "lower") -> float:
    """Deterministic, documented tie-breaking when scores are equal-ranked."""
    if not candidates:
        return 0.0
    return min(candidates) if rule == "lower" else max(candidates)


# ============================================================================
# OUTLIER HANDLING  (defined before results are reviewed)
# ============================================================================

def detect_outliers(scores: Sequence[float], method: str = "iqr") -> List[int]:
    """Return indices of outlier scores using the pre-declared method (IQR by default)."""
    values = [float(s) for s in scores]
    if method != "iqr" or len(values) < 4:
        return []
    ordered = sorted(values)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [i for i, v in enumerate(values) if v < lo or v > hi]


# ============================================================================
# AGGREGATION  (documented weighting + exclusions)
# ============================================================================

def aggregate(metric_scores: Dict[str, float], policy: ScoringPolicy) -> Dict[str, float]:
    """
    Aggregate per-metric scores into a single number under an explicit policy.

    Honours documented exclusions and weights; unweighted metrics get weight 1.0.
    Returns the weighted mean plus the effective weights used (for transparency).
    """
    included = {m: s for m, s in metric_scores.items() if m not in policy.exclusions}
    if not included:
        return {"aggregate": 0.0, "metrics_used": 0}

    total_weight = 0.0
    weighted_sum = 0.0
    used_weights = {}
    for metric, score in included.items():
        w = policy.weights.get(metric, 1.0)
        used_weights[metric] = w
        weighted_sum += w * score
        total_weight += w

    return {
        "aggregate": round(weighted_sum / total_weight, 4) if total_weight else 0.0,
        "metrics_used": len(included),
        "weights": used_weights,
        "excluded": list(policy.exclusions),
    }
