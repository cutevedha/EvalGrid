"""
tests/test_batch_judging.py: Tests for multi-rubric batched LLM judging.

The headline value prop here is **token reduction**: when a user evaluates with N
LLM-judge metrics, EvalGrid makes ONE LLM call per case instead of N. We need to
prove:

  1. The batching actually fires when ≥2 LLM-judge metrics are requested.
  2. The resulting per-metric scores match what individual calls would have given.
  3. The total token spend is meaningfully lower than the unbatched path.
  4. The opt-out (``batch_judging=False``) restores the per-metric call pattern.
  5. The batch parser is robust to malformed JSON and missing rubrics.
"""

import json
import threading
import time
from typing import Dict, List

import pytest

from core.schemas import TestCase
from evalgrid import CostTracker, JudgeClient, evaluate
from evalgrid.batch_judge import (
    BATCHABLE_RUBRICS,
    METRIC_TO_RUBRIC,
    _build_batch_prompt,
    _normalise_score,
    _parse_batch_response,
    batch_judge_score,
    extract_batchable_metrics,
)


# ============================================================================
# MOCK ADAPTERS — count calls and tokens precisely
# ============================================================================

class CountingMockAdapter:
    """Mock LLM adapter that counts calls and tracks token usage."""

    model = "mock-batch-judge"

    def __init__(self, response: str = '{"correctness": 4, "relevance": 5, "fluency": 4, "helpfulness": 5, "completeness": 4}'):
        self.response = response
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self._lock = threading.Lock()

    def generate_sync(self, prompt: str, **kwargs) -> str:
        with self._lock:
            self.calls += 1
            self.total_prompt_tokens   += len(prompt.split())
            self.total_response_tokens += len(self.response.split())
        return self.response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_sync(prompt)


class SinglyTunedMockAdapter:
    """Mock adapter that returns a different (single-rubric) response per call."""

    model = "single-rubric-mock"
    SINGLE_RESPONSE = "REASONING: Looks fine. SCORE: 4"

    def __init__(self):
        self.calls = 0
        self.total_prompt_tokens = 0
        self._lock = threading.Lock()

    def generate_sync(self, prompt: str, **kwargs) -> str:
        with self._lock:
            self.calls += 1
            self.total_prompt_tokens += len(prompt.split())
        return self.SINGLE_RESPONSE

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_sync(prompt)


# ============================================================================
# BATCHING DETECTION — extract_batchable_metrics
# ============================================================================

class TestExtractBatchableMetrics:
    def test_all_batchable_recognised(self):
        result = extract_batchable_metrics([
            "llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency",
        ])
        assert set(result) == {"llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency"}

    def test_non_batchable_excluded(self):
        # Reference-correctness and refusal_quality have different prompt structures.
        result = extract_batchable_metrics([
            "llm_judge_correctness",
            "llm_judge_reference_correctness",   # NOT batchable
            "refusal_quality",                    # NOT batchable
            "exact_match",                        # not even a judge metric
        ])
        assert result == ["llm_judge_correctness"]

    def test_empty_input(self):
        assert extract_batchable_metrics([]) == []

    def test_only_non_judge_metrics(self):
        assert extract_batchable_metrics(["exact_match", "substring_match"]) == []


# ============================================================================
# PROMPT CONSTRUCTION — fast structural assertions
# ============================================================================

class TestBuildBatchPrompt:
    def test_includes_all_rubrics(self):
        prompt = _build_batch_prompt(
            input_text="What is gravity?",
            output_text="Gravity attracts masses.",
            rubrics=["correctness", "relevance", "fluency"],
        )
        for rubric in ["correctness", "relevance", "fluency"]:
            assert rubric in prompt

    def test_includes_input_and_output(self):
        prompt = _build_batch_prompt(
            input_text="QUESTION_XYZ",
            output_text="ANSWER_ABC",
            rubrics=["correctness"],
        )
        assert "QUESTION_XYZ" in prompt
        assert "ANSWER_ABC" in prompt

    def test_includes_context_when_provided(self):
        prompt = _build_batch_prompt(
            input_text="Q",
            output_text="A",
            rubrics=["groundedness"],
            context="CONTEXT_HERE",
        )
        assert "CONTEXT_HERE" in prompt

    def test_omits_context_block_when_none(self):
        prompt = _build_batch_prompt(
            input_text="Q",
            output_text="A",
            rubrics=["correctness"],
            context=None,
        )
        assert "Context:" not in prompt

    def test_prompt_is_compact(self):
        # The whole prompt for 5 rubrics + a short Q/A should be reasonably small
        prompt = _build_batch_prompt(
            input_text="A short question.",
            output_text="A short response.",
            rubrics=["correctness", "relevance", "fluency", "helpfulness", "completeness"],
        )
        # < 250 words = roughly < 320 tokens for the whole multi-rubric prompt
        assert len(prompt.split()) < 250


# ============================================================================
# RESPONSE PARSING — robustness to malformed JSON
# ============================================================================

class TestParseBatchResponse:
    def test_clean_json(self):
        response = '{"correctness": 4, "relevance": 5}'
        scores = _parse_batch_response(response, ["correctness", "relevance"])
        assert scores["correctness"] == 0.8
        assert scores["relevance"] == 1.0

    def test_json_with_surrounding_text(self):
        response = 'Sure! Here are the scores:\n{"correctness": 3, "relevance": 4}\nLet me know if you want more detail.'
        scores = _parse_batch_response(response, ["correctness", "relevance"])
        assert scores["correctness"] == 0.6
        assert scores["relevance"] == 0.8

    def test_json_in_markdown_fence(self):
        response = '```json\n{"correctness": 5, "relevance": 5}\n```'
        scores = _parse_batch_response(response, ["correctness", "relevance"])
        assert scores["correctness"] == 1.0

    def test_missing_rubric_uses_neutral_midpoint(self):
        # Caller asked for "correctness" and "fluency" but the LLM only returned "correctness"
        response = '{"correctness": 4}'
        scores = _parse_batch_response(response, ["correctness", "fluency"])
        assert scores["correctness"] == 0.8
        assert scores["fluency"] == 0.6  # neutral midpoint

    def test_malformed_json_falls_back(self):
        # All scores fall back to the neutral midpoint
        scores = _parse_batch_response("not even close to JSON", ["correctness", "relevance"])
        assert all(s == 0.6 for s in scores.values())

    def test_empty_response_falls_back(self):
        scores = _parse_batch_response("", ["correctness"])
        assert scores["correctness"] == 0.6

    def test_case_insensitive_keys(self):
        response = '{"Correctness": 5, "RELEVANCE": 4}'
        scores = _parse_batch_response(response, ["correctness", "relevance"])
        assert scores["correctness"] == 1.0
        assert scores["relevance"] == 0.8


# ============================================================================
# NORMALISE SCORE
# ============================================================================

class TestNormaliseScore:
    def test_valid_range(self):
        assert _normalise_score(5) == 1.0
        assert _normalise_score(3) == 0.6
        assert _normalise_score(1) == 0.2

    def test_string_number(self):
        assert _normalise_score("4") == 0.8

    def test_none_falls_back(self):
        assert _normalise_score(None) == 0.6

    def test_out_of_range_falls_back(self):
        # Negative or absurdly large → neutral fallback
        assert _normalise_score(-2) == 0.6
        assert _normalise_score(999) == 0.6


# ============================================================================
# END-TO-END BATCH CALL
# ============================================================================

class TestBatchJudgeScore:
    def test_returns_score_per_rubric(self):
        adapter = CountingMockAdapter()
        from evalgrid import set_judge
        set_judge(JudgeClient(adapter))

        case = TestCase(id="x", project="p", capability="generation", input="Q?")
        scores = batch_judge_score(case, "An answer.", ["correctness", "relevance", "fluency"])
        assert set(scores.keys()) == {"correctness", "relevance", "fluency"}
        assert all(0 <= s <= 1 for s in scores.values())

    def test_one_llm_call_for_many_rubrics(self):
        adapter = CountingMockAdapter()
        from evalgrid import set_judge
        set_judge(JudgeClient(adapter))

        case = TestCase(id="x", project="p", capability="generation", input="Q?")
        batch_judge_score(case, "An answer.", ["correctness", "relevance", "fluency", "helpfulness", "completeness"])
        assert adapter.calls == 1  # ONE call covers all five rubrics

    def test_no_client_falls_back_to_heuristic(self):
        from evalgrid import set_judge
        set_judge(None)

        case = TestCase(id="x", project="p", capability="generation", input="Q?")
        scores = batch_judge_score(case, "A reasonable answer.", ["correctness", "relevance"])
        assert "correctness" in scores
        assert "relevance" in scores

    def test_empty_output_returns_zero(self):
        case = TestCase(id="x", project="p", capability="generation", input="Q?")
        scores = batch_judge_score(case, "", ["correctness", "relevance"])
        assert scores["correctness"] == 0.0
        assert scores["relevance"] == 0.0

    def test_non_batchable_rubrics_filtered_out(self):
        from evalgrid import set_judge
        set_judge(JudgeClient(CountingMockAdapter()))

        case = TestCase(id="x", project="p", capability="generation", input="Q?")
        # "not_a_real_rubric" is filtered out; only correctness scored
        scores = batch_judge_score(case, "An answer.", ["correctness", "not_a_real_rubric"])
        assert "correctness" in scores
        assert "not_a_real_rubric" not in scores


# ============================================================================
# EVALUATE() INTEGRATION — the headline value
# ============================================================================

class TestEvaluateBatchedJudging:
    def test_three_metrics_one_llm_call_per_case(self):
        """With 3 LLM-judge metrics on 10 cases, batching means 10 calls (not 30)."""
        adapter = CountingMockAdapter()
        judge = JudgeClient(adapter)
        cases = [{"id": f"c{i}", "input": f"q{i}", "output": f"a{i}"} for i in range(10)]
        evaluate(
            cases=cases,
            metrics=["llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency"],
            judge=judge,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        # 10 cases × 1 batched call each = 10 total LLM calls (not 30)
        assert adapter.calls == 10

    def test_batching_disabled_makes_n_calls(self):
        """With batching off, 3 metrics → 3 LLM calls per case."""
        adapter = SinglyTunedMockAdapter()
        judge = JudgeClient(adapter, cache=False)
        cases = [{"id": f"c{i}", "input": f"q{i}", "output": f"a{i}"} for i in range(5)]
        evaluate(
            cases=cases,
            metrics=["llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency"],
            judge=judge,
            batch_judging=False,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        # 5 cases × 3 metrics = 15 calls
        assert adapter.calls == 15

    def test_single_metric_not_batched(self):
        """1 LLM-judge metric: no batching (would just add overhead)."""
        adapter = SinglyTunedMockAdapter()
        judge = JudgeClient(adapter, cache=False)
        cases = [{"id": f"c{i}", "input": f"q{i}", "output": f"a{i}"} for i in range(5)]
        evaluate(
            cases=cases,
            metrics=["llm_judge_correctness"],
            judge=judge,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        # 5 cases × 1 metric, no batching = 5 calls
        assert adapter.calls == 5

    def test_batched_scores_distributed_correctly(self):
        """Verify each metric receives its own score from the batch."""
        adapter = CountingMockAdapter(
            response='{"correctness": 5, "relevance": 3, "fluency": 1}'
        )
        judge = JudgeClient(adapter)
        run = evaluate(
            cases=[{"id": "c1", "input": "Q?", "output": "A."}],
            metrics=["llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency"],
            judge=judge,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        scores = run.results[0].scores
        assert scores["llm_judge_correctness"] == 1.0
        assert scores["llm_judge_relevance"] == 0.6
        assert scores["llm_judge_fluency"] == 0.2

    def test_mixed_batchable_and_non_batchable(self):
        """Batchable metrics share a call; non-batchable metrics make their own."""
        adapter = SinglyTunedMockAdapter()
        judge = JudgeClient(adapter, cache=False)
        cases = [{"id": f"c{i}", "input": f"q{i}", "output": f"a{i}"} for i in range(3)]

        evaluate(
            cases=cases,
            metrics=[
                "llm_judge_correctness",   # batchable
                "llm_judge_relevance",     # batchable
                "exact_match",              # not LLM
            ],
            judge=judge,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        # 3 cases × 1 batched call = 3 LLM calls
        # exact_match is not an LLM metric so no calls from it
        assert adapter.calls == 3


# ============================================================================
# TOKEN REDUCTION — the marketing claim
# ============================================================================

class TestTokenReduction:
    def test_batched_uses_fewer_tokens_than_unbatched(self):
        """Marketing-grade headline: batched judging reduces total tokens.

        Concretely: with 5 LLM-judge metrics × 20 cases, batched should consume
        at least 50% fewer total prompt tokens than the per-metric approach.
        """
        # Per-metric (unbatched) baseline
        unbatched_adapter = SinglyTunedMockAdapter()
        evaluate(
            cases=[
                {"id": f"c{i}", "input": f"What is gravity, version {i}?",
                 "output": f"Gravity is a force that attracts masses, explanation {i}."}
                for i in range(20)
            ],
            metrics=[
                "llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency",
                "llm_judge_helpfulness", "llm_judge_completeness",
            ],
            judge=JudgeClient(unbatched_adapter, cache=False),
            batch_judging=False,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        unbatched_calls   = unbatched_adapter.calls
        unbatched_tokens  = unbatched_adapter.total_prompt_tokens

        # Batched
        batched_adapter = CountingMockAdapter(
            response='{"correctness": 4, "relevance": 4, "fluency": 4, "helpfulness": 4, "completeness": 4}'
        )
        evaluate(
            cases=[
                {"id": f"c{i}", "input": f"What is gravity, version {i}?",
                 "output": f"Gravity is a force that attracts masses, explanation {i}."}
                for i in range(20)
            ],
            metrics=[
                "llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency",
                "llm_judge_helpfulness", "llm_judge_completeness",
            ],
            judge=JudgeClient(batched_adapter, cache=False),
            batch_judging=True,
            concurrency=1,
            progress=False,
            quiet=True,
        )
        batched_calls  = batched_adapter.calls
        batched_tokens = batched_adapter.total_prompt_tokens

        # Calls: should be exactly 5x fewer
        assert batched_calls == 20  # 20 cases × 1 batched call
        assert unbatched_calls == 100  # 20 cases × 5 metrics

        # Tokens: batched should use less than HALF of unbatched
        savings_ratio = 1 - (batched_tokens / unbatched_tokens)
        assert savings_ratio >= 0.5, (
            f"Expected ≥50% token reduction; got {savings_ratio*100:.0f}%.\n"
            f"  unbatched: {unbatched_tokens} tokens across {unbatched_calls} calls\n"
            f"  batched:   {batched_tokens} tokens across {batched_calls} calls"
        )


# ============================================================================
# CONCURRENCY — batched judging must be thread-safe
# ============================================================================

class TestBatchingUnderConcurrency:
    def test_batched_judging_concurrent_no_score_mixup(self):
        """Each case must see ITS OWN batch scores, not another concurrent case's."""

        # Adapter that returns case-id-specific scores
        class CaseSpecificAdapter:
            model = "case-specific"
            def __init__(self):
                self.calls = 0
                self._lock = threading.Lock()
            def generate_sync(self, prompt, **kwargs):
                with self._lock:
                    self.calls += 1
                # Pluck the case-id substring out of the prompt and use it as the score
                if "case_high" in prompt:
                    return '{"correctness": 5, "relevance": 5, "fluency": 5}'
                if "case_low" in prompt:
                    return '{"correctness": 1, "relevance": 1, "fluency": 1}'
                return '{"correctness": 3, "relevance": 3, "fluency": 3}'
            async def generate(self, prompt, **kwargs):
                return self.generate_sync(prompt)

        adapter = CaseSpecificAdapter()
        judge = JudgeClient(adapter, cache=False)
        cases = [
            {"id": "high_1", "input": "case_high question 1", "output": "out 1"},
            {"id": "low_1",  "input": "case_low question 1",  "output": "out 2"},
            {"id": "high_2", "input": "case_high question 2", "output": "out 3"},
            {"id": "low_2",  "input": "case_low question 2",  "output": "out 4"},
        ] * 5

        run = evaluate(
            cases=cases,
            metrics=["llm_judge_correctness", "llm_judge_relevance", "llm_judge_fluency"],
            judge=judge,
            concurrency=10,
            progress=False,
            quiet=True,
        )

        # Every "high_*" case should score 1.0 and every "low_*" case should score 0.2
        for r in run.results:
            expected = 1.0 if r.test_id.startswith("high") else 0.2
            assert r.scores["llm_judge_correctness"] == expected, (
                f"Case {r.test_id} got mixed-up batch scores: {r.scores}"
            )
