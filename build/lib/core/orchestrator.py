"""
core/orchestrator.py: The central coordinator that runs every evaluation.

The Orchestrator is the brain of EvalGrid.  When you want to judge an AI's answer,
you call orchestrator.run(test_case, actual_output) and it automatically:

  1. Runs deterministic checks (exact match, keyword presence, regex patterns).
  2. Runs semantic checks (how similar is the answer in meaning?).
  3. Checks safety / policy compliance.
  4. Asks an LLM judge for a quality rating.
  5. Validates any required JSON structure.
  6. Detects PII (emails, phone numbers, credit cards) in the output.
  7. Detects prompt-injection attacks in the input.
  8. Decides PASS / FAIL based on the thresholds in the test case.

You can run a single evaluation (run / run_async) or a large batch in parallel
(run_batch / run_batch_async).  The concurrency parameter controls how many
evaluations run simultaneously so you don't overwhelm external APIs.
"""

from core.schemas import EvalResult, TestCase
from core.metric_registry import MetricRegistry
from evals.deterministic import evaluate as det_eval
from evals.semantic import evaluate as sem_eval
from evals.safety import evaluate as safe_eval
from evals.llm_judge import judge_score
from evals.json_schema import validate_json_output
from guards.pii import detect_pii, mask_pii
from guards.prompt_injection import is_prompt_injection
from typing import Dict, List, Optional, Any
import asyncio


# Bumped when the evaluation logic changes in a way that affects comparability of
# historical runs: makes it easy to filter results by evaluator generation.
EVALUATOR_VERSION = "1.0"


class Orchestrator:
    """
    Main orchestrator for running evaluations

    Coordinates all evaluation metrics, safety checks, and guards
    Supports both sync and async evaluation
    """

    def __init__(self, llm_client=None, embedder=None):
        """
        Initialize the orchestrator

        Args:
            llm_client: Optional LLM client for judge-based metrics
            embedder: Optional embedder for semantic metrics
        """
        self.llm_client = llm_client
        self.embedder = embedder
        self.metric_registry = MetricRegistry()

    def run(self, test_case: TestCase, actual_output: str) -> EvalResult:
        """
        Run evaluation synchronously (blocking)

        Args:
            test_case: The test case to evaluate
            actual_output: The actual output from the AI system

        Returns:
            EvalResult with all metric scores and pass/fail status
        """
        return asyncio.run(self.run_async(test_case, actual_output))

    async def run_async(self, test_case: TestCase, actual_output: str) -> EvalResult:
        """
        Run evaluation asynchronously (non-blocking)

        Evaluates the output using all applicable metrics:
        1. Deterministic metrics (exact match, patterns)
        2. Semantic metrics (similarity, embeddings)
        3. Safety metrics (policy compliance)
        4. LLM judge metrics (AI-powered evaluation)
        5. JSON schema validation
        6. PII detection and prompt injection detection

        Args:
            test_case: The test case to evaluate
            actual_output: The actual output from the AI system

        Returns:
            EvalResult with all metric scores and pass/fail status
        """
        scores = {}

        # Run deterministic evaluations (fast, no dependencies)
        scores.update(det_eval(test_case, actual_output))
        # Run semantic evaluations (similarity-based)
        scores.update(sem_eval(test_case, actual_output))
        # Run safety evaluations (policy compliance)
        scores.update(safe_eval(test_case, actual_output))

        # Run LLM judge evaluations (AI-powered, slower)
        scores["judge_correctness"] = judge_score(test_case.input, actual_output, "correctness", test_case.context)
        scores["judge_groundedness"] = judge_score(test_case.input, actual_output, "groundedness", test_case.context)

        # Validate JSON structure if expected
        if test_case.expected_json is not None:
            scores.update(validate_json_output(actual_output, list(test_case.expected_json.keys())))

        # Detect PII (Personally Identifiable Information)
        pii = detect_pii(actual_output)
        scores["pii_found"] = 1.0 if any(pii.values()) else 0.0
        # Detect prompt injection attempts
        scores["prompt_injection_detected"] = 1.0 if is_prompt_injection(test_case.input) else 0.0

        # Determine if test passed based on thresholds
        thresholds = test_case.thresholds or {"exact_match": 1.0, "policy_safe": 1.0}
        passed = all(scores.get(k, 0.0) >= v for k, v in thresholds.items()) and scores["pii_found"] == 0.0

        # Generate notes for failed tests
        notes = [] if passed else ["One or more thresholds failed", f"PII masked: {mask_pii(actual_output)}"]

        return EvalResult(
            test_id=test_case.id,
            passed=passed,
            scores=scores,
            notes=notes,
            evaluator_version=EVALUATOR_VERSION,
        )

    async def run_batch_async(self, test_cases: List[TestCase], outputs: Dict[str, str], concurrency: int = 5) -> List[EvalResult]:
        """
        Run batch evaluation asynchronously with concurrency control

        Evaluates multiple test cases in parallel for efficiency

        Args:
            test_cases: List of test cases to evaluate
            outputs: Dictionary mapping test_id to actual output
            concurrency: Maximum number of concurrent evaluations

        Returns:
            List of EvalResults
        """
        # Create semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(concurrency)

        async def run_with_semaphore(test_case: TestCase):
            """Run evaluation with semaphore to control concurrency"""
            async with semaphore:
                return await self.run_async(test_case, outputs.get(test_case.id, ""))

        # Run all evaluations concurrently
        tasks = [run_with_semaphore(tc) for tc in test_cases]
        return await asyncio.gather(*tasks)

    def run_batch(self, test_cases: List[TestCase], outputs: Dict[str, str], concurrency: int = 5) -> List[EvalResult]:
        """
        Run batch evaluation synchronously

        Args:
            test_cases: List of test cases to evaluate
            outputs: Dictionary mapping test_id to actual output
            concurrency: Maximum number of concurrent evaluations

        Returns:
            List of EvalResults
        """
        return asyncio.run(self.run_batch_async(test_cases, outputs, concurrency))

    def compute_metric(self, metric_name: str, test_case: TestCase, actual_output: str, **kwargs) -> Optional[float]:
        """
        Compute a single metric by name

        Args:
            metric_name: Name of the metric to compute
            test_case: The test case being evaluated
            actual_output: The actual output from the AI system
            **kwargs: Additional parameters for the metric

        Returns:
            Metric score or None if metric not found
        """
        return self.metric_registry.compute(metric_name, test_case, actual_output, **kwargs)

    def list_available_metrics(self, capability: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
        """
        List all available metrics with optional filtering

        Args:
            capability: Filter by AI capability (e.g., "agent", "rag")
            tag: Filter by tag (e.g., "safety", "custom")

        Returns:
            List of metric names
        """
        return self.metric_registry.list_metrics(tag=tag, capability=capability)

    def get_metric_metadata(self, metric_name: str):
        """
        Get metadata for a metric

        Args:
            metric_name: Name of the metric

        Returns:
            MetricMetadata object or None
        """
        return self.metric_registry.get_metadata(metric_name)

    async def run_with_custom_metrics(self, test_case: TestCase, actual_output: str, metric_names: List[str]) -> Dict[str, Any]:
        """
        Run evaluation with specific custom metrics

        Args:
            test_case: The test case to evaluate
            actual_output: The actual output from the AI system
            metric_names: List of metric names to compute

        Returns:
            Dictionary of metric_name -> score
        """
        scores = {}

        # Compute each requested metric
        for metric_name in metric_names:
            try:
                score = self.compute_metric(metric_name, test_case, actual_output)
                if score is not None:
                    scores[metric_name] = score
            except Exception as e:
                # Log error and continue with other metrics
                scores[metric_name] = 0.0

        return scores

