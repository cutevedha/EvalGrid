# Agent Memory - Tracks evaluation state across adaptive rounds
# The autonomous agent reflects on what it has already learned to decide where to dig
# deeper, so it needs a running record of which probe each result came from and how
# each probe is scoring. Memory is the single source of truth the reflection step reads.

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from core.schemas import EvalResult, TestCase

from agent.planner import ProbeSpec
from agent.report import ProbeFinding


# ============================================================================
# AGENT MEMORY
# ============================================================================

class AgentMemory:
    """
    Accumulates test cases and their results, indexed by the probe that produced them.

    Provides the aggregates the reflection step needs: per-probe pass rates, the
    weakest gate metric, and sample failing inputs to mutate into harder cases.
    """

    def __init__(self):
        # probe_name -> list of (TestCase, EvalResult)
        self._by_probe: Dict[str, List[Tuple[TestCase, EvalResult]]] = defaultdict(list)
        # probe_name -> ProbeSpec (so we keep severity, capability, gate metrics)
        self._specs: Dict[str, ProbeSpec] = {}

    # ------------------------------------------------------------------
    # RECORDING
    # ------------------------------------------------------------------

    def record(self, probe: ProbeSpec, test_case: TestCase, result: EvalResult) -> None:
        """Store a single evaluated case under its probe."""
        self._specs[probe.name] = probe
        self._by_probe[probe.name].append((test_case, result))

    def all_results(self) -> List[EvalResult]:
        """Every EvalResult recorded so far, across all probes and rounds."""
        return [res for pairs in self._by_probe.values() for _, res in pairs]

    def seen_inputs(self, probe_name: str) -> set:
        """Inputs already tried for a probe — used to avoid re-running duplicates."""
        return {tc.input for tc, _ in self._by_probe.get(probe_name, [])}

    # ------------------------------------------------------------------
    # AGGREGATION / REFLECTION SUPPORT
    # ------------------------------------------------------------------

    def finding_for(self, probe_name: str) -> ProbeFinding:
        """
        Roll up everything recorded for a probe into a single ProbeFinding.

        The gate score is the mean across the probe's gate metrics, normalised so that
        "lower is better" metrics (e.g. prompt_injection_detected with threshold 0.0)
        contribute correctly — a case passes its gate when every gate metric clears.
        """
        spec = self._specs[probe_name]
        pairs = self._by_probe[probe_name]
        cases_run = len(pairs)

        if cases_run == 0:
            return ProbeFinding(
                probe=probe_name, capability=spec.capability, severity=spec.severity,
                cases_run=0, pass_rate=1.0, mean_gate_score=1.0,
            )

        passed = sum(1 for _, res in pairs if res.passed)

        # Track per-metric means so we can name the weakest gate metric.
        metric_totals: Dict[str, float] = defaultdict(float)
        for _, res in pairs:
            for metric, threshold in spec.gate_metrics.items():
                metric_totals[metric] += _gate_component(res.scores.get(metric, 0.0), threshold)

        metric_means = {m: total / cases_run for m, total in metric_totals.items()}
        mean_gate_score = (
            sum(metric_means.values()) / len(metric_means) if metric_means else float(passed) / cases_run
        )

        weakest_metric, weakest_score = "", 1.0
        if metric_means:
            weakest_metric = min(metric_means, key=metric_means.get)
            weakest_score = metric_means[weakest_metric]

        failing_inputs = [tc.input for tc, res in pairs if not res.passed][:5]

        return ProbeFinding(
            probe=probe_name,
            capability=spec.capability,
            severity=spec.severity,
            cases_run=cases_run,
            pass_rate=passed / cases_run,
            mean_gate_score=mean_gate_score,
            weakest_metric=weakest_metric,
            weakest_metric_score=weakest_score,
            failing_inputs=failing_inputs,
        )

    def all_findings(self) -> List[ProbeFinding]:
        """One ProbeFinding per probe that has recorded at least one case."""
        return [self.finding_for(name) for name in self._by_probe]

    def spec(self, probe_name: str) -> ProbeSpec:
        """Return the stored ProbeSpec for a probe name."""
        return self._specs[probe_name]


# ============================================================================
# HELPERS
# ============================================================================

def _gate_component(score: float, threshold: float) -> float:
    """
    Normalise a single metric's contribution to a 0–1 'goodness' value.

    For a "higher is better" gate (threshold > 0) the raw score already represents
    goodness. For a "must be zero" gate such as prompt_injection_detected (threshold
    0.0) we invert it so 0.0 -> 1.0 (good) and 1.0 -> 0.0 (bad).
    """
    if threshold <= 0.0:
        return 1.0 - min(max(score, 0.0), 1.0)
    return min(max(score, 0.0), 1.0)
