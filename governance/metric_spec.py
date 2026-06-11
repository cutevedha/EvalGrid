# Metric Validity - Category 3
# A written definition + rationale for every metric, task-alignment tracking, uncertainty
# estimates (bootstrap confidence intervals), and inter-rater agreement for human labels.
# The catalog cross-checks the live MetricRegistry so undefined metrics are surfaced, not
# silently trusted.

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ============================================================================
# METRIC SPECIFICATION
# ============================================================================

@dataclass
class MetricSpec:
    """
    The documented contract for a metric — the antidote to "easy to compute but invalid".

    Fields force the author to state not just *what* the metric computes but *why* it is
    the right measure for the task and whether it is deterministic.
    """
    name: str
    definition: str                      # What the score means
    rationale: str                       # Why it is aligned to the task (not just convenient)
    task_alignment: List[str] = field(default_factory=list)  # Capabilities it is valid for
    deterministic: bool = True
    higher_is_better: bool = True
    uncertainty_method: str = "bootstrap_ci"

    def validate(self) -> None:
        if not self.definition.strip() or not self.rationale.strip():
            raise ValueError(f"Metric '{self.name}' must have a definition and a rationale")


class MetricCatalog:
    """
    Holds MetricSpecs and checks them against the registered metrics.

    ``coverage`` reports which live metrics still lack a written definition — directly
    supporting the spec rule "every metric has a written definition and rationale".
    """

    def __init__(self):
        self._specs: Dict[str, MetricSpec] = {}

    def register(self, spec: MetricSpec) -> None:
        spec.validate()
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[MetricSpec]:
        return self._specs.get(name)

    def coverage(self, registered_metric_names: Sequence[str]) -> Dict[str, object]:
        """Compare documented specs against the live registry."""
        documented = set(self._specs)
        live = set(registered_metric_names)
        undocumented = sorted(live - documented)
        return {
            "documented": len(documented & live),
            "total_live": len(live),
            "coverage_rate": round(len(documented & live) / len(live), 3) if live else 0.0,
            "undocumented": undocumented,
        }


# ============================================================================
# UNCERTAINTY  (confidence intervals)
# ============================================================================

def bootstrap_ci(scores: Sequence[float], confidence: float = 0.95,
                 iterations: int = 1000, seed: int = 0) -> Dict[str, float]:
    """
    Bootstrap confidence interval for a mean score — the "error bar" the spec demands.

    Returns mean plus the lower/upper bounds of the central ``confidence`` interval.
    Deterministic given ``seed`` so the same scores always yield the same interval.
    """
    values = [float(s) for s in scores]
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    if n == 1:
        return {"mean": values[0], "lower": values[0], "upper": values[0], "n": 1}

    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - confidence) / 2 * iterations)
    hi_idx = int((1 + confidence) / 2 * iterations) - 1
    return {
        "mean": round(sum(values) / n, 4),
        "lower": round(means[lo_idx], 4),
        "upper": round(means[max(hi_idx, 0)], 4),
        "n": n,
    }


# ============================================================================
# INTER-RATER AGREEMENT  (when humans or judges label)
# ============================================================================

def cohen_kappa(rater_a: Sequence, rater_b: Sequence) -> float:
    """
    Cohen's kappa for two raters labelling the same items (categorical agreement beyond chance).

    Returns a value in [-1, 1]; 1 is perfect agreement, 0 is chance-level.
    """
    if len(rater_a) != len(rater_b) or not rater_a:
        return 0.0
    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    labels = set(rater_a) | set(rater_b)
    expected = 0.0
    for label in labels:
        pa = sum(1 for x in rater_a if x == label) / n
        pb = sum(1 for x in rater_b if x == label) / n
        expected += pa * pb

    if expected == 1.0:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def agreement_rate(rater_a: Sequence, rater_b: Sequence) -> float:
    """Simple proportion of items where two raters agree."""
    if len(rater_a) != len(rater_b) or not rater_a:
        return 0.0
    return round(sum(1 for a, b in zip(rater_a, rater_b) if a == b) / len(rater_a), 4)
