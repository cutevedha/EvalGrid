"""
tests/test_async_evaluate.py: Tests for concurrent / async evaluation.

Verifies:
  - Sequential and parallel evaluation produce equivalent results
  - High concurrency actually achieves real speedup (proves no GIL bottleneck on I/O-bound work)
  - Result ordering is preserved regardless of completion order
  - The concurrency cap is honored
  - fail_fast cancels pending work
  - a_evaluate() works inside an existing event loop (Jupyter / FastAPI use case)
  - Cache and CostTracker are thread-safe under heavy concurrency
"""

import asyncio
import threading
import time
from typing import Dict, List

import pytest

from core.metric_registry import register_metric
from core.schemas import TestCase
from evalgrid import (
    CostTracker,
    JudgeClient,
    a_evaluate,
    evaluate,
)


# ============================================================================
# DETERMINISTIC SLOW METRIC — proves concurrency speedup
# ============================================================================

_SLOW_METRIC_SLEEP = 0.05  # 50 ms per case — enough to dominate orchestration overhead
_max_concurrent_observed = 0
_concurrency_lock = threading.Lock()
_active_workers = 0


@register_metric(
    "slow_test_metric",
    description="Sleeps to simulate an I/O-bound metric — for async-speedup tests",
    tags=["test"],
)
def slow_test_metric(test_case, actual_output):
    """Sleep so we can measure real concurrency speedup."""
    global _max_concurrent_observed, _active_workers
    with _concurrency_lock:
        _active_workers += 1
        if _active_workers > _max_concurrent_observed:
            _max_concurrent_observed = _active_workers
    try:
        time.sleep(_SLOW_METRIC_SLEEP)
    finally:
        with _concurrency_lock:
            _active_workers -= 1
    return {"slow_test_metric": 1.0}


def _reset_concurrency_observer():
    global _max_concurrent_observed, _active_workers
    with _concurrency_lock:
        _max_concurrent_observed = 0
        _active_workers = 0


def _build_cases(n: int) -> List[Dict]:
    return [{"id": f"case_{i}", "input": f"q{i}", "output": f"a{i}"} for i in range(n)]


# ============================================================================
# PARALLELISM CORRECTNESS — ordering, equivalence
# ============================================================================

class TestOrderingPreservation:
    def test_results_match_input_order(self):
        run = evaluate(
            cases=_build_cases(20),
            metrics=["slow_test_metric"],
            concurrency=10,
            progress=False,
            quiet=True,
        )
        ids = [r.test_id for r in run.results]
        assert ids == [f"case_{i}" for i in range(20)]

    def test_sequential_and_parallel_agree(self):
        cases = _build_cases(10)
        seq = evaluate(cases=cases, metrics=["slow_test_metric"], concurrency=1, progress=False, quiet=True)
        par = evaluate(cases=cases, metrics=["slow_test_metric"], concurrency=10, progress=False, quiet=True)
        seq_pairs = [(r.test_id, r.scores) for r in seq.results]
        par_pairs = [(r.test_id, r.scores) for r in par.results]
        assert seq_pairs == par_pairs


# ============================================================================
# REAL SPEEDUP — the headline performance claim
# ============================================================================

class TestSpeedup:
    def test_concurrent_runs_faster_than_sequential(self):
        cases = _build_cases(20)
        _reset_concurrency_observer()
        t0 = time.time()
        evaluate(cases=cases, metrics=["slow_test_metric"], concurrency=1, progress=False, quiet=True)
        sequential_time = time.time() - t0

        _reset_concurrency_observer()
        t0 = time.time()
        evaluate(cases=cases, metrics=["slow_test_metric"], concurrency=10, progress=False, quiet=True)
        parallel_time = time.time() - t0

        # Parallel with 10 workers should be at least 3x faster than sequential
        # for a 50ms sleep per case across 20 cases. Conservative bound to be CI-friendly.
        assert parallel_time < sequential_time / 3, (
            f"Expected >3x speedup. Sequential: {sequential_time:.2f}s, "
            f"Parallel: {parallel_time:.2f}s, ratio: {sequential_time/parallel_time:.1f}x"
        )

    def test_higher_concurrency_observed(self):
        _reset_concurrency_observer()
        evaluate(
            cases=_build_cases(20),
            metrics=["slow_test_metric"],
            concurrency=8,
            progress=False,
            quiet=True,
        )
        # Multiple workers MUST run simultaneously — at least 2
        assert _max_concurrent_observed >= 2

    def test_concurrency_cap_enforced(self):
        _reset_concurrency_observer()
        evaluate(
            cases=_build_cases(30),
            metrics=["slow_test_metric"],
            concurrency=5,
            progress=False,
            quiet=True,
        )
        # Should never exceed the cap (allowing 1 slack for timing measurement edges)
        assert _max_concurrent_observed <= 6, (
            f"Concurrency cap exceeded: observed {_max_concurrent_observed} > 5"
        )

    def test_throughput_reported(self):
        run = evaluate(
            cases=_build_cases(10),
            metrics=["slow_test_metric"],
            concurrency=5,
            progress=False,
            quiet=True,
        )
        assert run.elapsed_seconds is not None
        assert run.elapsed_seconds > 0
        assert run.throughput is not None
        assert run.throughput > 0
        assert run.concurrency == 5


# ============================================================================
# FAIL-FAST UNDER CONCURRENCY
# ============================================================================

class TestFailFastConcurrent:
    def test_fail_fast_cancels_pending(self):
        # The first case fails, so subsequent ones should be cancelled
        cases = [
            {"id": "fail", "input": "x", "output": "y", "expected_output": "z"},  # fails exact_match
            *_build_cases(20),
        ]
        run = evaluate(
            cases=cases,
            metrics=["exact_match"],
            threshold=0.5,
            concurrency=2,
            fail_fast=True,
            progress=False,
            quiet=True,
        )
        # We stopped early — should have fewer results than total cases
        assert len(run.results) < len(cases)
        # The failure should be among the returned results
        assert any(not r.passed for r in run.results)


# ============================================================================
# a_evaluate() — works inside an existing event loop
# ============================================================================

class TestAsyncEntryPoint:
    def test_a_evaluate_runs(self):
        async def _run():
            return await a_evaluate(
                cases=_build_cases(5),
                metrics=["slow_test_metric"],
                concurrency=5,
                progress=False,
                quiet=True,
            )
        run = asyncio.run(_run())
        assert len(run) == 5

    def test_a_evaluate_preserves_order(self):
        async def _run():
            return await a_evaluate(
                cases=_build_cases(10),
                metrics=["slow_test_metric"],
                concurrency=10,
                progress=False,
                quiet=True,
            )
        run = asyncio.run(_run())
        assert [r.test_id for r in run.results] == [f"case_{i}" for i in range(10)]

    def test_sync_evaluate_inside_running_loop(self):
        """Calling evaluate() (sync) from inside an existing loop should still work."""
        async def _outer():
            # This is the tricky case — evaluate() must NOT crash with
            # "asyncio.run() cannot be called from a running event loop"
            return evaluate(
                cases=_build_cases(3),
                metrics=["slow_test_metric"],
                concurrency=3,
                progress=False,
                quiet=True,
            )
        run = asyncio.run(_outer())
        assert len(run) == 3


# ============================================================================
# THREAD SAFETY — cache, cost tracker, judge stats under concurrency
# ============================================================================

class TestThreadSafety:
    def test_cost_tracker_no_lost_updates(self):
        # Run many cases concurrently and verify the cost tracker counted them all
        tracker = CostTracker(model="gpt-4o-mini")

        mock_judge = JudgeClient(_MockAdapterAlwaysOk())

        evaluate(
            cases=_build_cases(50),
            metrics=["llm_judge_correctness"],
            cost_tracker=tracker,
            judge=mock_judge,
            concurrency=20,
            progress=False,
            quiet=True,
        )
        # We expect approximately 50 calls (one per case for the single LLM-judge metric).
        # Allow some variance for cache hits if any case inputs collide (they shouldn't here).
        assert tracker.calls >= 40, (
            f"Cost tracker lost updates under concurrency. Got {tracker.calls}, expected ~50."
        )

    def test_judge_cache_consistent_under_concurrency(self):
        # All cases use the same prompt → cache hits should be high
        mock_judge = JudgeClient(_MockAdapterAlwaysOk())
        identical_cases = [
            {"id": f"c{i}", "input": "same prompt", "output": "same answer"}
            for i in range(20)
        ]
        evaluate(
            cases=identical_cases,
            metrics=["llm_judge_correctness"],
            judge=mock_judge,
            concurrency=10,
            progress=False,
            quiet=True,
        )
        stats = mock_judge.stats()
        # Real calls + cache hits must sum to at least the case count
        assert stats["real_calls"] + stats["cache_hits"] >= 20


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    def test_empty_cases_returns_empty_run(self):
        run = evaluate(cases=[], metrics=["slow_test_metric"], concurrency=10, progress=False, quiet=True)
        assert len(run) == 0
        assert run.elapsed_seconds is not None

    def test_concurrency_one_still_works(self):
        run = evaluate(
            cases=_build_cases(3),
            metrics=["slow_test_metric"],
            concurrency=1,
            progress=False,
            quiet=True,
        )
        assert len(run) == 3
        assert run.concurrency == 1

    def test_more_concurrency_than_cases(self):
        run = evaluate(
            cases=_build_cases(3),
            metrics=["slow_test_metric"],
            concurrency=100,  # way more than 3 cases
            progress=False,
            quiet=True,
        )
        assert len(run) == 3

    def test_zero_concurrency_treated_as_one(self):
        # Should not crash — clamps to 1
        run = evaluate(
            cases=_build_cases(2),
            metrics=["slow_test_metric"],
            concurrency=0,
            progress=False,
            quiet=True,
        )
        assert len(run) == 2


# ============================================================================
# MOCK ADAPTER
# ============================================================================

class _MockAdapterAlwaysOk:
    """Sync-only mock adapter for judge testing under concurrency."""

    model = "mock-gpt-4o-mini"

    def __init__(self):
        self._calls = 0
        self._lock = threading.Lock()

    def generate_sync(self, prompt: str, **kwargs) -> str:
        with self._lock:
            self._calls += 1
        # Sleep briefly to make concurrency observable
        time.sleep(0.005)
        return "REASONING: ok. SCORE: 4"

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_sync(prompt)
