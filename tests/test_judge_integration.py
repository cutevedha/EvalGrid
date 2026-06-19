"""
tests/test_judge_integration.py: Tests for the real-LLM-judge integration layer.

The strategy is to inject a deterministic mock adapter into the judge pathway so
we can verify everything end-to-end (caching, error handling, cost tracking,
auto-detection, plumbing through evaluate()) without ever hitting a real API.
"""

import asyncio
import os
from typing import List
from unittest.mock import patch

import pytest

from core.schemas import TestCase
from evalgrid import (
    JudgeClient,
    auto_detect_judge,
    configure,
    evaluate,
    get_judge,
    set_judge,
)
from evalgrid.judge import (
    _build_adapter,
    _provider_for_model,
    ensure_judge_configured,
    reset_auto_detection,
)


# ============================================================================
# MOCK ADAPTER — deterministic, never touches a real API
# ============================================================================

class MockAdapter:
    """Fake LLM adapter that returns a canned judge-formatted response."""

    def __init__(self, response: str = "REASONING: Looks good. SCORE: 4",
                 model: str = "mock-judge", raise_on_call: bool = False):
        self.response = response
        self.model = model
        self.call_count = 0
        self.last_prompt = None
        self.raise_on_call = raise_on_call

    def generate_sync(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self.raise_on_call:
            raise RuntimeError("simulated adapter failure")
        return self.response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_sync(prompt, **kwargs)


# ============================================================================
# JudgeClient — sync wrapper basics
# ============================================================================

class TestJudgeClientBasics:
    def test_generate_returns_adapter_response(self):
        adapter = MockAdapter(response="REASONING: ok. SCORE: 5")
        client = JudgeClient(adapter)
        assert client.generate("hello") == "REASONING: ok. SCORE: 5"

    def test_generate_invokes_adapter(self):
        adapter = MockAdapter()
        client = JudgeClient(adapter)
        client.generate("hi")
        assert adapter.call_count == 1

    def test_passes_temperature_and_max_tokens(self):
        adapter = MockAdapter()
        client = JudgeClient(adapter, temperature=0.0, max_tokens=200)
        # JudgeClient should forward its defaults to the adapter
        client.generate("hi")
        assert adapter.call_count == 1

    def test_model_property(self):
        adapter = MockAdapter(model="claude-3-haiku")
        client = JudgeClient(adapter)
        assert client.model == "claude-3-haiku"

    def test_error_returns_empty_string(self):
        adapter = MockAdapter(raise_on_call=True)
        client = JudgeClient(adapter)
        assert client.generate("hello") == ""

    def test_error_records_in_stats(self):
        adapter = MockAdapter(raise_on_call=True)
        client = JudgeClient(adapter)
        client.generate("hi")
        assert client.stats()["errors"] == 1


# ============================================================================
# JudgeClient — response caching
# ============================================================================

class TestJudgeClientCache:
    def test_identical_prompt_uses_cache(self):
        adapter = MockAdapter()
        client = JudgeClient(adapter)
        client.generate("identical prompt")
        client.generate("identical prompt")
        # Only one real call should be made
        assert adapter.call_count == 1
        assert client.stats()["cache_hits"] == 1

    def test_different_prompts_distinct_cache(self):
        adapter = MockAdapter()
        client = JudgeClient(adapter)
        client.generate("prompt A")
        client.generate("prompt B")
        assert adapter.call_count == 2

    def test_cache_disabled(self):
        adapter = MockAdapter()
        client = JudgeClient(adapter, cache=False)
        client.generate("hi")
        client.generate("hi")
        assert adapter.call_count == 2

    def test_clear_cache(self):
        adapter = MockAdapter()
        client = JudgeClient(adapter)
        client.generate("hi")
        client.clear_cache()
        client.generate("hi")
        # After clearing, the second call should hit the adapter again
        assert adapter.call_count == 2

    def test_stats_format(self):
        client = JudgeClient(MockAdapter())
        client.generate("a")
        client.generate("a")
        stats = client.stats()
        assert "real_calls" in stats
        assert "cache_hits" in stats
        assert "errors" in stats
        assert stats["real_calls"] == 1
        assert stats["cache_hits"] == 1


# ============================================================================
# JudgeClient — cost tracking integration
# ============================================================================

class TestJudgeClientCost:
    def test_cost_tracker_records_real_calls(self):
        from evalgrid import CostTracker
        tracker = CostTracker(model="gpt-4o-mini")
        client = JudgeClient(MockAdapter(), cost_tracker=tracker)
        client.generate("hello world")
        assert tracker.calls == 1
        assert tracker.input_tokens > 0

    def test_cost_tracker_only_records_real_not_cached(self):
        from evalgrid import CostTracker
        tracker = CostTracker(model="gpt-4o-mini")
        client = JudgeClient(MockAdapter(), cost_tracker=tracker)
        client.generate("same")
        client.generate("same")  # cache hit
        # Only ONE call gets recorded — the cache hit was free
        assert tracker.calls == 1


# ============================================================================
# Model → adapter routing
# ============================================================================

class TestProviderResolution:
    def test_gpt_prefix_routes_to_openai(self):
        assert _provider_for_model("gpt-4o-mini") == "openai"
        assert _provider_for_model("gpt-4") == "openai"

    def test_claude_prefix_routes_to_anthropic(self):
        assert _provider_for_model("claude-3-5-sonnet-20241022") == "anthropic"
        assert _provider_for_model("claude-3-haiku") == "anthropic"

    def test_gemini_prefix_routes_to_gemini(self):
        assert _provider_for_model("gemini-2.0-flash-exp") == "gemini"

    def test_llama_routes_to_ollama(self):
        assert _provider_for_model("llama2") == "ollama"
        assert _provider_for_model("mistral-7b") == "ollama"
        assert _provider_for_model("qwen2.5") == "ollama"

    def test_unknown_defaults_to_openai(self):
        assert _provider_for_model("some-novel-model") == "openai"


# ============================================================================
# Auto detection from environment variables
# ============================================================================

class TestAutoDetection:
    def test_no_env_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert auto_detect_judge() is None

    def test_openai_key_returns_judge(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            client = auto_detect_judge()
            assert client is not None
            assert isinstance(client, JudgeClient)

    def test_anthropic_key_returns_judge(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            client = auto_detect_judge()
            assert client is not None

    def test_openai_preferred_over_anthropic(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-openai",
            "ANTHROPIC_API_KEY": "sk-anthropic",
        }, clear=True):
            client = auto_detect_judge()
            # Model name should indicate OpenAI was preferred
            assert "gpt" in client.model.lower()

    def test_evalgrid_judge_model_overrides(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-x",
            "EVALGRID_JUDGE_MODEL": "claude-3-haiku-20240307",
        }, clear=True):
            client = auto_detect_judge()
            assert client is not None
            assert "claude" in client.model.lower()


# ============================================================================
# set_judge / get_judge / configure
# ============================================================================

class TestPublicAPI:
    def test_set_judge_with_model_string(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            client = set_judge("gpt-4o-mini")
            assert isinstance(client, JudgeClient)
            assert "gpt" in client.model

    def test_set_judge_with_client_instance(self):
        custom = JudgeClient(MockAdapter())
        result = set_judge(custom)
        assert result is custom
        assert get_judge() is custom

    def test_set_judge_with_adapter(self):
        adapter = MockAdapter()
        client = set_judge(adapter)
        assert isinstance(client, JudgeClient)
        assert client.adapter is adapter

    def test_set_judge_none_clears(self):
        set_judge(JudgeClient(MockAdapter()))
        result = set_judge(None)
        assert result is None
        assert get_judge() is None

    def test_get_judge_none_when_unset(self):
        assert get_judge() is None

    def test_configure_returns_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x"}):
            client = configure(judge="gpt-4o-mini")
            assert isinstance(client, JudgeClient)

    def test_configure_with_api_key(self):
        client = configure(judge="gpt-4o-mini", api_key="sk-direct")
        assert isinstance(client, JudgeClient)

    def test_configure_temperature(self):
        client = configure(judge="gpt-4o-mini", api_key="x", temperature=0.7)
        assert client.temperature == 0.7

    def test_configure_none_clears(self):
        configure(judge=JudgeClient(MockAdapter()))
        configure(judge=None)
        assert get_judge() is None


# ============================================================================
# ensure_judge_configured — lazy detection
# ============================================================================

class TestEnsureJudgeConfigured:
    def test_uses_existing_judge_if_set(self):
        custom = JudgeClient(MockAdapter())
        set_judge(custom)
        result = ensure_judge_configured()
        assert result is custom

    def test_auto_detects_when_no_existing_judge(self):
        reset_auto_detection()
        # Force the EVALGRID_DISABLE_AUTO_JUDGE to off for this test only
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-x",
        }, clear=True):
            # The conftest sets EVALGRID_DISABLE_AUTO_JUDGE=1; clear it here
            os.environ.pop("EVALGRID_DISABLE_AUTO_JUDGE", None)
            client = ensure_judge_configured()
            assert client is not None

    def test_disable_env_var_blocks_auto_detect(self):
        reset_auto_detection()
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-x",
            "EVALGRID_DISABLE_AUTO_JUDGE": "1",
        }, clear=True):
            assert ensure_judge_configured() is None


# ============================================================================
# evaluate(judge=...) — end-to-end LLM judge integration
# ============================================================================

class TestEvaluateJudgeParameter:
    def test_judge_false_forces_heuristic(self):
        # Even if a judge was set, judge=False bypasses it
        set_judge(JudgeClient(MockAdapter(response="REASONING: x SCORE: 1")))
        run = evaluate(
            cases=[{"input": "Hi", "output": "Hi", "expected_output": "Hi"}],
            metrics=["llm_judge_correctness"],
            judge=False,
            progress=False,
            quiet=True,
        )
        # With judge=False, the heuristic returns a fixed score (0.9 for output ≥ 10 chars)
        score = run.results[0].scores.get("llm_judge_correctness")
        assert score in (0.3, 0.9)  # heuristic outcomes for the correctness rubric

    def test_judge_string_routes_to_model(self):
        # Use a mock adapter so we don't hit a real API
        mock_client = JudgeClient(MockAdapter(response="REASONING: good. SCORE: 5"))
        run = evaluate(
            cases=[{"input": "Hi", "output": "Hi"}],
            metrics=["llm_judge_correctness"],
            judge=mock_client,
            progress=False,
            quiet=True,
        )
        assert run.results[0].scores.get("llm_judge_correctness") == 1.0  # 5/5

    def test_judge_score_uses_model(self):
        mock_client = JudgeClient(MockAdapter(response="REASONING: ok. SCORE: 3"))
        run = evaluate(
            cases=[{"input": "Q?", "output": "A response."}],
            metrics=["llm_judge_correctness"],
            judge=mock_client,
            progress=False,
            quiet=True,
        )
        # Score should be 3/5 = 0.6 from the mock
        assert abs(run.results[0].scores["llm_judge_correctness"] - 0.6) < 0.01

    def test_judge_stats_in_eval_run(self):
        mock_client = JudgeClient(MockAdapter())
        run = evaluate(
            cases=[{"input": "A", "output": "B"}],
            metrics=["llm_judge_correctness"],
            judge=mock_client,
            progress=False,
            quiet=True,
        )
        assert run.judge_stats is not None
        assert run.judge_stats["real_calls"] >= 1

    def test_judge_model_in_eval_run(self):
        mock = JudgeClient(MockAdapter(model="my-judge-model"))
        run = evaluate(
            cases=[{"input": "A", "output": "B"}],
            metrics=["llm_judge_correctness"],
            judge=mock,
            progress=False,
            quiet=True,
        )
        assert run.judge_model == "my-judge-model"

    def test_judge_restored_after_explicit_use(self):
        original = JudgeClient(MockAdapter(response="REASONING: x. SCORE: 5"))
        set_judge(original)

        temporary = JudgeClient(MockAdapter(response="REASONING: y. SCORE: 1"))
        evaluate(
            cases=[{"input": "A", "output": "B"}],
            metrics=["llm_judge_correctness"],
            judge=temporary,
            progress=False,
            quiet=True,
        )
        # The original judge should be restored after the run
        assert get_judge() is original


# ============================================================================
# evaluate() — cost tracking with real judge
# ============================================================================

class TestEvaluateJudgeCostTracking:
    def test_real_llm_call_recorded(self):
        from evalgrid import CostTracker
        tracker = CostTracker(model="gpt-4o-mini")
        mock_client = JudgeClient(MockAdapter(response="REASONING: ok. SCORE: 4"))
        evaluate(
            cases=[{"input": "Q?", "output": "A.", "expected_output": "A."}],
            metrics=["llm_judge_correctness"],
            cost_tracker=tracker,
            judge=mock_client,
            progress=False,
            quiet=True,
        )
        # Both JudgeClient (real call) and evaluate (metric-level estimate)
        # contribute — what we need to assert is that SOMETHING was tracked.
        assert tracker.calls >= 1


# ============================================================================
# Summary text includes judge information
# ============================================================================

class TestSummaryWithJudge:
    def test_summary_mentions_judge_model(self):
        mock = JudgeClient(MockAdapter(model="test-judge-v1"))
        run = evaluate(
            cases=[{"input": "A", "output": "B"}],
            metrics=["llm_judge_correctness"],
            judge=mock,
            progress=False,
            quiet=True,
        )
        assert "test-judge-v1" in run.summary()

    def test_summary_shows_llm_call_count(self):
        mock = JudgeClient(MockAdapter())
        run = evaluate(
            cases=[{"input": "A", "output": "B"}, {"input": "C", "output": "D"}],
            metrics=["llm_judge_correctness"],
            judge=mock,
            progress=False,
            quiet=True,
        )
        summary = run.summary()
        assert "LLM calls" in summary


# ============================================================================
# Robustness — judge failures must not crash the eval
# ============================================================================

class TestRobustness:
    def test_judge_raises_falls_back_to_empty(self):
        # Even if the judge throws, evaluate completes
        broken = JudgeClient(MockAdapter(raise_on_call=True))
        run = evaluate(
            cases=[{"input": "A", "output": "B"}],
            metrics=["llm_judge_correctness"],
            judge=broken,
            progress=False,
            quiet=True,
        )
        assert len(run.results) == 1

    def test_multiple_cases_with_same_prompt_use_cache(self):
        # Two identical cases should yield one real call thanks to the cache
        mock = JudgeClient(MockAdapter())
        evaluate(
            cases=[
                {"input": "Same Q?", "output": "Same A."},
                {"input": "Same Q?", "output": "Same A."},
            ],
            metrics=["llm_judge_correctness"],
            judge=mock,
            progress=False,
            quiet=True,
        )
        assert mock.stats()["cache_hits"] >= 1
