"""
pipelines/batch_runner.py: High-throughput and CI/CD evaluation pipelines.

This module provides three runner classes that sit on top of the Orchestrator
and handle how evaluations are executed at scale:

BatchRunner
-----------
Evaluates a large set of test cases in parallel using asyncio.  Uses a semaphore
to cap the number of simultaneous evaluations so external APIs are not overwhelmed.
After the run, it exposes summary statistics (pass rate, timing, score distribution).

StreamingRunner
---------------
Evaluates test cases one at a time and fires a callback after each result.  Ideal
for real-time dashboards, progress bars, or early-stopping logic where you want to
react to each result as it arrives rather than waiting for the whole batch.

GatingRunner
------------
CI/CD quality gate evaluator.  You define named gates, each linking a metric to a
minimum acceptable average score.  After evaluation, gates report pass/fail and the
runner exposes all_gates_passed(): call sys.exit(1) if False to block a deployment.

    Example::
        gating = GatingRunner(orchestrator)
        gating.add_gate("safety_gate", "policy_safe", threshold=1.0, severity="critical")
        gate_results = gating.evaluate_gates_sync(results)
        if not gating.all_gates_passed():
            sys.exit(1)  # Block the release
"""

from core.schemas import TestCase, EvalResult
from core.orchestrator import Orchestrator
from typing import List, Dict, Any, Optional, Callable
import asyncio
import time


# ============================================================================
# BATCH RUNNER
# ============================================================================

class BatchRunner:
    """
    Runs a large collection of test cases in parallel using asyncio.

    Controls concurrency with a semaphore so the system is not overwhelmed.
    Collects summary metrics (pass rate, timing) after the run.
    """

    def __init__(self, orchestrator: Orchestrator, concurrency: int = 5):
        """
        Args:
            orchestrator: Configured Orchestrator instance
            concurrency: Maximum number of evaluations running simultaneously
        """
        self.orchestrator = orchestrator
        self.concurrency = concurrency
        self.results = []  # Stored after run for later querying
        self.metrics = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "total_time_ms": 0,
            "avg_time_per_test_ms": 0,
        }

    async def run_batch_async(self, test_cases: List[TestCase], outputs: Dict[str, str]) -> List[EvalResult]:
        """
        Evaluate all test cases concurrently and record timing metrics.

        Args:
            test_cases: List of test cases to evaluate
            outputs: Dict mapping test_id -> actual AI output

        Returns:
            List of EvalResults in the same order as test_cases
        """
        start_time = time.time()

        results = await self.orchestrator.run_batch_async(test_cases, outputs, self.concurrency)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        self.results = results
        self._update_metrics(results, total_time_ms)

        return results

    def run_batch(self, test_cases: List[TestCase], outputs: Dict[str, str]) -> List[EvalResult]:
        """Synchronous wrapper around run_batch_async"""
        return asyncio.run(self.run_batch_async(test_cases, outputs))

    def _update_metrics(self, results: List[EvalResult], total_time_ms: float) -> None:
        """Compute and store summary statistics after a batch run"""
        self.metrics["total_tests"]         = len(results)
        self.metrics["passed_tests"]        = sum(1 for r in results if r.passed)
        self.metrics["failed_tests"]        = len(results) - self.metrics["passed_tests"]
        self.metrics["total_time_ms"]       = total_time_ms
        self.metrics["avg_time_per_test_ms"] = total_time_ms / len(results) if results else 0

    def get_metrics(self) -> Dict[str, Any]:
        """Return summary statistics from the last batch run"""
        return self.metrics

    def get_pass_rate(self) -> float:
        """Fraction of test cases that passed (0.0-1.0)"""
        if self.metrics["total_tests"] == 0:
            return 0.0
        return self.metrics["passed_tests"] / self.metrics["total_tests"]

    def get_results_by_severity(self, severity: str) -> List[EvalResult]:
        """Filter stored results by severity prefix in the test ID"""
        return [r for r in self.results if r.test_id.startswith(severity)]

    def get_failed_tests(self) -> List[EvalResult]:
        """Return only the results that did not pass"""
        return [r for r in self.results if not r.passed]

    def get_passed_tests(self) -> List[EvalResult]:
        """Return only the results that passed"""
        return [r for r in self.results if r.passed]

    def filter_results(self, predicate: Callable[[EvalResult], bool]) -> List[EvalResult]:
        """
        Return results matching an arbitrary predicate function.

        Args:
            predicate: Callable(EvalResult) -> bool
        """
        return [r for r in self.results if predicate(r)]

    def aggregate_scores(self, metric_name: str) -> Dict[str, Any]:
        """
        Compute descriptive statistics for a single metric across all results.

        Args:
            metric_name: Name of the metric to aggregate

        Returns:
            Dict with count, mean, min, max, sum
        """
        scores = [r.scores.get(metric_name, 0.0) for r in self.results if metric_name in r.scores]

        if not scores:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}

        return {
            "count": len(scores),
            "mean":  sum(scores) / len(scores),
            "min":   min(scores),
            "max":   max(scores),
            "sum":   sum(scores),
        }

    def get_score_distribution(self, metric_name: str, bins: int = 10) -> Dict[str, int]:
        """
        Bucket metric scores into a histogram.

        Args:
            metric_name: Name of the metric to analyse
            bins: Number of histogram buckets

        Returns:
            Dict mapping bucket label -> count
        """
        scores = [r.scores.get(metric_name, 0.0) for r in self.results if metric_name in r.scores]

        if not scores:
            return {}

        min_score = min(scores)
        max_score = max(scores)
        bin_width = (max_score - min_score) / bins if max_score > min_score else 1

        distribution = {}
        for score in scores:
            bin_idx = int((score - min_score) / bin_width) if bin_width > 0 else 0
            bin_idx = min(bin_idx, bins - 1)
            bin_label = f"{min_score + bin_idx * bin_width:.2f}-{min_score + (bin_idx + 1) * bin_width:.2f}"
            distribution[bin_label] = distribution.get(bin_label, 0) + 1

        return distribution


# ============================================================================
# STREAMING RUNNER
# ============================================================================

class StreamingRunner:
    """
    Evaluates test cases one at a time and fires a callback after each result.

    Useful for real-time dashboards, progress bars, or early-stopping logic.
    """

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.results = []

    async def run_streaming(self, test_cases: List[TestCase], outputs: Dict[str, str], callback: Optional[Callable] = None) -> List[EvalResult]:
        """
        Evaluate each test case sequentially and invoke the callback after each one.

        Args:
            test_cases: Ordered list of test cases
            outputs: Dict mapping test_id -> actual AI output
            callback: Optional async or sync callable(EvalResult) called after each evaluation

        Returns:
            All EvalResults in order
        """
        for test_case in test_cases:
            result = await self.orchestrator.run_async(test_case, outputs.get(test_case.id, ""))
            self.results.append(result)

            if callback:
                await callback(result)  # Notify caller of each completed result

        return self.results

    def run_streaming_sync(self, test_cases: List[TestCase], outputs: Dict[str, str], callback: Optional[Callable] = None) -> List[EvalResult]:
        """Synchronous wrapper around run_streaming"""
        return asyncio.run(self.run_streaming(test_cases, outputs, callback))


# ============================================================================
# GATING RUNNER
# ============================================================================

class GatingRunner:
    """
    CI/CD quality gate evaluator.

    Each gate maps a metric to a minimum acceptable average score.
    After evaluation, gates report pass/fail status: failed gates can block deployments.

    Example usage::
        gating = GatingRunner(orchestrator)
        gating.add_gate("safety_gate", "policy_safe", threshold=1.0, severity="critical")
        gating.add_gate("quality_gate", "llm_judge_correctness", threshold=0.8)
        gate_results = gating.evaluate_gates_sync(results)
        if not gating.all_gates_passed():
            sys.exit(1)  # Block the deployment
    """

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.gates = {}  # gate_name -> gate config dict

    def add_gate(self, gate_name: str, metric_name: str, threshold: float, severity: str = "medium") -> None:
        """
        Register a quality gate.

        Args:
            gate_name: Unique identifier for this gate
            metric_name: Metric to evaluate (must be in EvalResult.scores)
            threshold: Minimum acceptable average score (0.0-1.0)
            severity: Informational severity label (low/medium/high/critical)
        """
        self.gates[gate_name] = {
            "metric":    metric_name,
            "threshold": threshold,
            "severity":  severity,
            "passed":    False,  # Updated after evaluate_gates()
        }

    async def evaluate_gates(self, results: List[EvalResult]) -> Dict[str, bool]:
        """
        Check all gates against a batch of evaluation results.

        A gate passes when the average metric score across all results
        meets or exceeds its threshold.

        Args:
            results: List of EvalResults from a batch run

        Returns:
            Dict mapping gate_name -> bool (True = passed)
        """
        gate_results = {}

        for gate_name, gate_config in self.gates.items():
            metric_name = gate_config["metric"]
            threshold   = gate_config["threshold"]

            # Collect scores for this metric across all results
            scores = [r.scores.get(metric_name, 0.0) for r in results if metric_name in r.scores]

            if not scores:
                gate_results[gate_name] = False  # No data: fail the gate
                continue

            avg_score = sum(scores) / len(scores)
            gate_results[gate_name] = avg_score >= threshold
            self.gates[gate_name]["passed"] = gate_results[gate_name]

        return gate_results

    def evaluate_gates_sync(self, results: List[EvalResult]) -> Dict[str, bool]:
        """Synchronous wrapper around evaluate_gates"""
        return asyncio.run(self.evaluate_gates(results))

    def get_gate_status(self) -> Dict[str, Any]:
        """Return the full gate config including pass/fail status for all gates"""
        return self.gates

    def all_gates_passed(self) -> bool:
        """True only when every registered gate has passed"""
        return all(gate["passed"] for gate in self.gates.values())
