# Acceptance Criteria - Category 10
# Pass/fail thresholds set in advance, a minimum sample size before a benchmark counts,
# stricter gates for critical metrics than exploratory ones, a rule that no release rests
# on a single metric, and an explicit split between "measurement confidence" and "model
# quality". Complements pipelines/GatingRunner (which averages a metric across results) by
# adding the policy layer on top.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# GATE DEFINITION
# ============================================================================

TIER_CRITICAL = "critical"
TIER_EXPLORATORY = "exploratory"


@dataclass
class Gate:
    """A single acceptance gate, declared *before* the run."""
    metric: str
    threshold: float
    tier: str = TIER_EXPLORATORY         # critical gates block release; exploratory inform
    direction: str = "min"               # "min" -> score must be >= threshold; "max" -> <=

    def passes(self, value: float) -> bool:
        return value >= self.threshold if self.direction == "min" else value <= self.threshold


# ============================================================================
# ACCEPTANCE POLICY
# ============================================================================

@dataclass
class AcceptancePolicy:
    """
    A pre-registered acceptance policy. Thresholds and sample size are fixed in advance so
    they cannot be moved to fit results.
    """
    gates: List[Gate] = field(default_factory=list)
    min_sample_size: int = 30
    require_multiple_metrics: bool = True   # no release based on a single metric alone

    def add_gate(self, metric: str, threshold: float, tier: str = TIER_EXPLORATORY,
                 direction: str = "min") -> "AcceptancePolicy":
        self.gates.append(Gate(metric=metric, threshold=threshold, tier=tier, direction=direction))
        return self

    def evaluate(self, metric_means: Dict[str, float], sample_size: int,
                 confidence_ok: bool = True) -> Dict[str, Any]:
        """
        Decide acceptance from pre-set gates and aggregated metric means.

        Critical gate failures block release; exploratory failures are reported but do not
        block. Separately reports measurement confidence (sample size + caller-supplied
        confidence flag) so "we couldn't measure well" is never confused with "the model
        is bad".
        """
        results = []
        for gate in self.gates:
            value = metric_means.get(gate.metric)
            present = value is not None
            ok = present and gate.passes(value)
            results.append({
                "metric": gate.metric, "tier": gate.tier, "threshold": gate.threshold,
                "value": value, "present": present, "passed": ok,
            })

        critical_failures = [r for r in results if r["tier"] == TIER_CRITICAL and not r["passed"]]
        exploratory_failures = [r for r in results if r["tier"] == TIER_EXPLORATORY and not r["passed"]]
        gates_evaluated = len(results)

        # Measurement confidence is independent of model quality.
        enough_samples = sample_size >= self.min_sample_size
        single_metric_violation = self.require_multiple_metrics and gates_evaluated < 2

        measurement_confident = enough_samples and confidence_ok and not single_metric_violation
        model_quality_pass = not critical_failures

        # Release requires BOTH adequate measurement AND passing critical gates.
        accepted = measurement_confident and model_quality_pass

        reasons = []
        if not enough_samples:
            reasons.append(f"sample size {sample_size} < required {self.min_sample_size}")
        if single_metric_violation:
            reasons.append("release would rest on a single metric; >=2 gates required")
        if not confidence_ok:
            reasons.append("measurement confidence flagged low by caller")
        for r in critical_failures:
            reasons.append(f"critical gate failed: {r['metric']} (value={r['value']}, need {r['threshold']})")

        return {
            "accepted": accepted,
            "model_quality_pass": model_quality_pass,        # distinct from...
            "measurement_confident": measurement_confident,  # ...measurement confidence
            "critical_failures": critical_failures,
            "exploratory_failures": exploratory_failures,
            "gate_results": results,
            "reasons": reasons,
        }
