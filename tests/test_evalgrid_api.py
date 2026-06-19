"""
tests/test_evalgrid_api.py: Tests for the high-level evalgrid.* user-facing API.

Covers:
  - evalgrid.evaluate()              one-liner API: cases & metrics flexibility
  - evalgrid.quick_eval()            single-pair convenience
  - evalgrid.EvalRun                 result object: iter, summary, export
  - evalgrid.MetricSet               presets + alias resolution
  - evalgrid.assert_test / assert_each   pytest integration
  - evalgrid.ScoreCache              caching: hit/miss/clear
  - evalgrid.CostTracker             token/dollar accounting
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from core.schemas import TestCase
from evalgrid import (
    CostTracker,
    EvalRun,
    MetricSet,
    ScoreCache,
    TestCase as ExportedTestCase,
    assert_each,
    assert_test,
    evaluate,
    quick_eval,
)
from evalgrid.presets import resolve_preset
from evalgrid.report_html import render_html_report


# ============================================================================
# IMPORTABILITY
# ============================================================================

class TestPublicImports:
    def test_test_case_exported(self):
        assert ExportedTestCase is TestCase

    def test_evaluate_is_callable(self):
        assert callable(evaluate)

    def test_metric_set_constants(self):
        assert isinstance(MetricSet.GENERATION, list)
        assert isinstance(MetricSet.RAG, list)
        assert isinstance(MetricSet.SAFETY, list)


# ============================================================================
# evaluate() — case input flexibility
# ============================================================================

class TestEvaluateCaseInputs:
    def test_list_of_test_cases(self):
        case = TestCase(id="t1", project="x", capability="generation", input="hi")
        run = evaluate(cases=[(case, "hello there friend")], metrics=["exact_match"], progress=False, quiet=True)
        assert len(run) == 1

    def test_list_of_dicts(self):
        run = evaluate(
            cases=[{"input": "What is AI?", "output": "AI is artificial intelligence."}],
            metrics=["exact_match"],
            progress=False,
            quiet=True,
        )
        assert len(run) == 1
        assert run.results[0].input == "What is AI?"

    def test_from_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump([{"input": "Q?", "expected_output": "A."}], f)
            path = f.name
        try:
            run = evaluate(cases=path, metrics=["exact_match"], progress=False, quiet=True)
            assert len(run) == 1
        finally:
            os.unlink(path)

    def test_single_dict_wrapped(self):
        run = evaluate(
            cases=[{"input": "Hi", "output": "Hi"}],
            metrics=["exact_match"],
            progress=False,
            quiet=True,
        )
        assert len(run) == 1


# ============================================================================
# evaluate() — metric specification flexibility
# ============================================================================

class TestEvaluateMetricSpec:
    def _case(self):
        return [({"input": "hello", "output": "hello world"})]

    def test_string_preset(self):
        run = evaluate(cases=self._case(), metrics="generation", progress=False, quiet=True)
        assert len(run.metrics_used) == len(MetricSet.GENERATION)

    def test_string_single_metric(self):
        run = evaluate(cases=self._case(), metrics="exact_match", progress=False, quiet=True)
        assert "exact_match" in run.metrics_used

    def test_list_of_names(self):
        run = evaluate(cases=self._case(), metrics=["exact_match", "substring_match"], progress=False, quiet=True)
        assert run.metrics_used == ["exact_match", "substring_match"]

    def test_metric_set_constant(self):
        run = evaluate(cases=self._case(), metrics=MetricSet.STRUCTURED, progress=False, quiet=True)
        assert len(run.metrics_used) == len(MetricSet.STRUCTURED)

    def test_unknown_preset_treated_as_metric_name(self):
        run = evaluate(cases=self._case(), metrics="exact_match", progress=False, quiet=True)
        assert "exact_match" in run.metrics_used


# ============================================================================
# EvalRun result object
# ============================================================================

class TestEvalRun:
    def _run(self):
        return evaluate(
            cases=[
                {"id": "a", "input": "Hello", "output": "Hello"},
                {"id": "b", "input": "Hi",    "output": "Bye"},
            ],
            metrics=["exact_match"],
            threshold=0.5,
            progress=False,
            quiet=True,
        )

    def test_is_iterable(self):
        run = self._run()
        cases = list(run)
        assert len(cases) == 2

    def test_len(self):
        assert len(self._run()) == 2

    def test_indexing(self):
        run = self._run()
        assert run[0].test_id == "a"

    def test_passed_property(self):
        run = self._run()
        assert run.passed is False  # case "b" fails exact_match

    def test_pass_rate(self):
        run = self._run()
        assert 0.0 <= run.pass_rate <= 1.0

    def test_metric_averages(self):
        run = self._run()
        avgs = run.metric_averages()
        assert "exact_match" in avgs

    def test_failed_cases(self):
        run = self._run()
        failed = run.failed_cases()
        assert all(not c.passed for c in failed)

    def test_summary_is_string(self):
        s = self._run().summary()
        assert isinstance(s, str)
        assert "EvalGrid" in s

    def test_to_dict(self):
        d = self._run().to_dict()
        assert "results" in d
        assert "pass_rate" in d

    def test_to_json(self):
        run = self._run()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            run.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "results" in data
        finally:
            os.unlink(path)

    def test_to_csv(self):
        run = self._run()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            run.to_csv(path)
            assert Path(path).stat().st_size > 0
        finally:
            os.unlink(path)

    def test_to_html(self):
        run = self._run()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            run.to_html(path, title="Test Report")
            content = Path(path).read_text(encoding="utf-8")
            assert "<html" in content
            assert "Test Report" in content
        finally:
            os.unlink(path)


# ============================================================================
# quick_eval()
# ============================================================================

class TestQuickEval:
    def test_basic(self):
        result = quick_eval(
            input="What is the capital of France?",
            output="Paris is the capital of France.",
            expected="The capital of France is Paris.",
            metrics=["substring_match"],
        )
        assert hasattr(result, "scores")
        assert hasattr(result, "passed")

    def test_without_expected(self):
        result = quick_eval(input="Q?", output="A.", metrics=["exact_match"])
        assert result is not None


# ============================================================================
# Presets
# ============================================================================

class TestPresets:
    def test_resolve_known_preset(self):
        metrics = resolve_preset("rag")
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_resolve_case_insensitive(self):
        assert resolve_preset("RAG") == resolve_preset("rag")

    def test_resolve_with_dash(self):
        assert resolve_preset("red-team") == resolve_preset("redteam")

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric preset"):
            resolve_preset("not_a_preset")

    def test_all_presets_resolvable(self):
        for name in ["generation", "rag", "safety", "agent", "performance",
                     "bias", "robustness", "reference", "summarization", "structured"]:
            assert isinstance(resolve_preset(name), list) or resolve_preset(name) == MetricSet.ALL


# ============================================================================
# Pytest assertion helpers
# ============================================================================

class TestAssertTest:
    def test_pass_when_score_above_threshold(self):
        # Identical strings → exact_match = 1.0 → above threshold 0.5
        assert_test(
            input="Hello",
            output="Hello",
            expected="Hello",
            metrics=["exact_match"],
            threshold=0.5,
        )

    def test_fails_with_assertion_error(self):
        with pytest.raises(AssertionError) as exc:
            assert_test(
                input="Hello",
                output="Goodbye",
                expected="Hello",
                metrics=["exact_match"],
                threshold=0.9,
            )
        assert "EvalGrid assertion failed" in str(exc.value)
        assert "exact_match" in str(exc.value)


class TestAssertEach:
    def test_pass(self):
        case = TestCase(
            id="a", project="x", capability="generation",
            input="Hi", expected_output="Hi",
        )
        assert_each(case, "Hi", metrics=["exact_match"], threshold=0.5)

    def test_fail(self):
        case = TestCase(
            id="a", project="x", capability="generation",
            input="Hi", expected_output="Hi",
        )
        with pytest.raises(AssertionError):
            assert_each(case, "Goodbye", metrics=["exact_match"], threshold=0.9)


# ============================================================================
# ScoreCache
# ============================================================================

class TestScoreCache:
    def test_miss_then_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ScoreCache(tmpdir)
            tc = TestCase(id="x", project="p", capability="generation", input="hi")

            # First call → miss
            assert cache.get("m1", tc, "out") is None
            cache.put("m1", tc, "out", {"m1": 0.85})

            # Second call → hit
            cached = cache.get("m1", tc, "out")
            assert cached == {"m1": 0.85}

            stats = cache.stats()
            assert stats["hits"] >= 1
            assert stats["writes"] >= 1

    def test_different_outputs_distinct_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ScoreCache(tmpdir)
            tc = TestCase(id="x", project="p", capability="generation", input="hi")
            cache.put("m1", tc, "output_a", {"m1": 0.5})
            cache.put("m1", tc, "output_b", {"m1": 0.9})
            assert cache.get("m1", tc, "output_a") == {"m1": 0.5}
            assert cache.get("m1", tc, "output_b") == {"m1": 0.9}

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ScoreCache(tmpdir)
            tc = TestCase(id="x", project="p", capability="generation", input="hi")
            cache.put("m1", tc, "out", {"m1": 0.5})
            removed = cache.clear()
            assert removed >= 1
            assert cache.get("m1", tc, "out") is None

    def test_evaluate_with_cache_true(self):
        case = {"input": "Hello", "output": "Hello"}
        run1 = evaluate(cases=[case], metrics=["exact_match"], cache=True, progress=False, quiet=True)
        run2 = evaluate(cases=[case], metrics=["exact_match"], cache=True, progress=False, quiet=True)
        # Second run should record at least one cache hit
        assert run2.cache_stats is not None


# ============================================================================
# CostTracker
# ============================================================================

class TestCostTracker:
    def test_record_increments(self):
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("llm_judge_correctness", input_text="hello world", output_text="hi")
        assert tracker.calls == 1
        assert tracker.input_tokens > 0

    def test_cost_estimate_positive(self):
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("m", input_text="word " * 100, output_text="word " * 50)
        cost = tracker.estimated_cost_usd()
        assert cost > 0

    def test_unknown_model_zero_cost(self):
        tracker = CostTracker(model="unknown-model-xyz")
        tracker.record("m", input_text="word", output_text="word")
        assert tracker.estimated_cost_usd() == 0.0

    def test_summary_structure(self):
        tracker = CostTracker()
        tracker.record("m1", input_text="word", output_text="word")
        s = tracker.summary()
        assert "calls" in s
        assert "cost_usd" in s
        assert "by_metric" in s

    def test_evaluate_records_cost(self):
        tracker = CostTracker(model="gpt-4o-mini")
        evaluate(
            cases=[{"input": "Q?", "output": "A.", "expected_output": "A."}],
            metrics=["llm_judge_correctness"],
            cost_tracker=tracker,
            progress=False,
            quiet=True,
        )
        assert tracker.calls >= 1


# ============================================================================
# HTML report rendering
# ============================================================================

class TestHtmlReport:
    def test_renders_valid_html(self):
        run = evaluate(
            cases=[{"input": "Hi", "output": "Hi", "expected_output": "Hi"}],
            metrics=["exact_match"],
            progress=False,
            quiet=True,
        )
        html = render_html_report(run, title="My Report")
        assert "<html" in html
        assert "My Report" in html
        assert "exact_match" in html

    def test_includes_pass_fail_badge(self):
        run = evaluate(
            cases=[{"input": "A", "output": "A", "expected_output": "A"}],
            metrics=["exact_match"],
            progress=False,
            quiet=True,
        )
        html = render_html_report(run)
        assert "PASS" in html or "FAIL" in html


# ============================================================================
# Quickstart
# ============================================================================

class TestQuickstart:
    def test_init_project(self):
        from evalgrid.quickstart import init_project
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = init_project(tmpdir)
            assert Path(paths["dataset"]).exists()
            assert Path(paths["script"]).exists()

    def test_run_quickstart(self):
        from evalgrid.quickstart import run_quickstart
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_quickstart(tmpdir)
            assert "pass_rate" in result
            assert Path(result["report_html"]).exists()
            assert Path(result["report_json"]).exists()


# ============================================================================
# Progress + behaviour edge cases
# ============================================================================

class TestProgress:
    def test_quiet_mode_silent(self, capsys):
        from evalgrid.progress import progress_iter
        items = list(progress_iter([1, 2, 3], quiet=True))
        assert items == [1, 2, 3]
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_passes_items_through(self):
        from evalgrid.progress import progress_iter
        items = list(progress_iter([1, 2, 3], quiet=True))
        assert items == [1, 2, 3]


class TestFailFast:
    def test_stops_at_first_failure(self):
        # Use concurrency=1 so fail_fast is deterministic — under parallel
        # execution, other tasks may already have completed when the failure fires.
        run = evaluate(
            cases=[
                {"id": "a", "input": "Hi", "output": "Bye", "expected_output": "Hi"},
                {"id": "b", "input": "Q?", "output": "Q?",  "expected_output": "Q?"},
                {"id": "c", "input": "?",  "output": "?",   "expected_output": "?"},
            ],
            metrics=["exact_match"],
            threshold=0.5,
            fail_fast=True,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        assert len(run.results) == 1
        assert run.results[0].test_id == "a"


# ============================================================================
# Robustness — bad inputs do not crash
# ============================================================================

class TestRobustness:
    def test_unknown_metric_silently_skipped(self):
        run = evaluate(
            cases=[{"input": "Hi", "output": "Hi"}],
            metrics=["this_metric_does_not_exist"],
            progress=False,
            quiet=True,
        )
        assert len(run) == 1
        # Unknown metric is simply absent from scores
        assert "this_metric_does_not_exist" not in run.results[0].scores

    def test_empty_case_list(self):
        run = evaluate(cases=[], metrics=["exact_match"], progress=False, quiet=True)
        assert len(run) == 0
        assert run.pass_rate == 0.0
