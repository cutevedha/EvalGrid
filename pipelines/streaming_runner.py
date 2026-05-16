# Streaming Runner - Real-time evaluation with per-result callbacks
# Evaluates test cases one at a time and fires a callback after each result
# Useful for live dashboards, progress tracking, and early-stopping logic

from core.schemas import TestCase, EvalResult
from core.orchestrator import Orchestrator
from typing import List, Dict, Optional, Callable
import asyncio


class StreamingRunner:
    """
    Evaluates test cases sequentially and notifies a callback after each one.

    Unlike BatchRunner (which waits for all results), StreamingRunner yields
    results one by one so callers can react immediately — useful for:
    - Real-time progress bars
    - Live dashboards
    - Early stopping when a critical test fails
    - Logging results as they arrive
    """

    def __init__(self, orchestrator: Orchestrator):
        """
        Args:
            orchestrator: Configured Orchestrator instance to run evaluations
        """
        self.orchestrator = orchestrator
        self.results: List[EvalResult] = []  # Accumulated results across all runs

    async def run_streaming(
        self,
        test_cases: List[TestCase],
        outputs: Dict[str, str],
        callback: Optional[Callable] = None,
    ) -> List[EvalResult]:
        """
        Evaluate test cases one by one, calling the callback after each result.

        Supports both async and sync callbacks — the runner detects which type
        is provided and calls it appropriately.

        Args:
            test_cases: Ordered list of test cases to evaluate
            outputs: Dict mapping test_id → actual AI output string
            callback: Optional callable(EvalResult) invoked after each evaluation.
                      May be async (coroutine function) or plain sync function.

        Returns:
            All EvalResults in the same order as test_cases
        """
        for test_case in test_cases:
            # Evaluate one test case at a time
            result = await self.orchestrator.run_async(
                test_case, outputs.get(test_case.id, "")
            )
            self.results.append(result)

            # Fire the callback if provided
            if callback:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)   # Async callback
                    else:
                        callback(result)          # Sync callback
                except Exception as e:
                    # Log but don't crash — callback errors are non-fatal
                    print(f"Callback error: {e}")

        return self.results

    def run_streaming_sync(
        self,
        test_cases: List[TestCase],
        outputs: Dict[str, str],
        callback: Optional[Callable] = None,
    ) -> List[EvalResult]:
        """
        Synchronous wrapper around run_streaming.

        Use this when you cannot use async/await in your calling code.
        """
        return asyncio.run(self.run_streaming(test_cases, outputs, callback))

    # ========================================================================
    # RESULT ACCESSORS
    # ========================================================================

    def get_results(self) -> List[EvalResult]:
        """Return all accumulated evaluation results"""
        return self.results

    def get_passed_count(self) -> int:
        """Number of test cases that passed"""
        return sum(1 for r in self.results if r.passed)

    def get_failed_count(self) -> int:
        """Number of test cases that failed"""
        return sum(1 for r in self.results if not r.passed)

    def get_pass_rate(self) -> float:
        """Fraction of test cases that passed (0.0–1.0)"""
        if not self.results:
            return 0.0
        return self.get_passed_count() / len(self.results)
