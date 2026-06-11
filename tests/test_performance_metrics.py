"""
Tests for the new performance, quality, reliability, agent-behavior, and RAG@k metrics.
Each test exercises one metric in isolation so regressions are easy to pinpoint.
"""

import pytest
from core.schemas import TestCase, AgentTrace, AgentStep, ToolCall, RAGTestCase


def _tc(capability="generation"):
    return TestCase(id="t1", project="p", capability=capability, input="Q?")


def _rag_tc(docs=None):
    return RAGTestCase(
        id="r1", project="p", capability="rag", input="Q?",
        documents=docs or ["the cat sat on the mat"],
        expected_output="cat sat mat",
    )


def _trace(steps=2, success=True, tool_errors=0):
    tool_calls = []
    if tool_errors:
        tool_calls = [ToolCall(name="search", actual_result="Error: not found")] * tool_errors
    s = [AgentStep(step_number=i + 1, action=f"act{i}", tool_calls=tool_calls) for i in range(steps)]
    return AgentTrace(agent_id="a1", steps=s, success=success)


# ============================================================================
# THROUGHPUT BREAKDOWN
# ============================================================================

def test_prefill_tokens_per_second_basic():
    from evals.performance_evals import prefill_tokens_per_second
    result = prefill_tokens_per_second(_tc(), "", input_tokens=1000, prefill_time_ms=500)
    assert result["prefill_tokens_per_second"] == pytest.approx(2000.0)


def test_prefill_tokens_per_second_missing_data():
    from evals.performance_evals import prefill_tokens_per_second
    assert prefill_tokens_per_second(_tc(), "")["prefill_tokens_per_second"] == 0.0


def test_input_tokens_per_request():
    from evals.performance_evals import input_tokens_per_request
    assert input_tokens_per_request(_tc(), "", input_tokens=512)["input_tokens_per_request"] == 512.0


def test_output_tokens_per_request():
    from evals.performance_evals import output_tokens_per_request
    assert output_tokens_per_request(_tc(), "", output_tokens=128)["output_tokens_per_request"] == 128.0


def test_cache_hit_rate():
    from evals.performance_evals import cache_hit_rate
    result = cache_hit_rate(_tc(), "", cache_hits=80, total_requests=100)
    assert result["cache_hit_rate"] == pytest.approx(0.8)


def test_cache_hit_rate_zero_requests():
    from evals.performance_evals import cache_hit_rate
    assert cache_hit_rate(_tc(), "", cache_hits=5, total_requests=0)["cache_hit_rate"] == 0.0


def test_cost_per_successful_task_all_pass():
    from evals.performance_evals import cost_per_successful_task
    result = cost_per_successful_task(_tc(), "", total_cost=1.0, successful_tasks=10, total_tasks=10)
    assert result["cost_per_successful_task"] == pytest.approx(1.0)


def test_cost_per_successful_task_half_pass():
    from evals.performance_evals import cost_per_successful_task
    result = cost_per_successful_task(_tc(), "", total_cost=1.0, successful_tasks=5, total_tasks=10)
    assert result["cost_per_successful_task"] == pytest.approx(0.5)


# ============================================================================
# QUALITY SIGNALS
# ============================================================================

def test_task_success_rate():
    from evals.performance_evals import task_success_rate
    result = task_success_rate(_tc(), "", successes=9, total_evals=10)
    assert result["task_success_rate"] == pytest.approx(0.9)


def test_task_success_rate_zero_evals():
    from evals.performance_evals import task_success_rate
    assert task_success_rate(_tc(), "")["task_success_rate"] == 0.0


def test_hallucination_rate_fully_grounded():
    from evals.performance_evals import hallucination_rate
    result = hallucination_rate(_tc(), "cat sat mat", context="the cat sat on the mat")
    assert result["hallucination_rate"] == pytest.approx(1.0)


def test_hallucination_rate_no_overlap():
    from evals.performance_evals import hallucination_rate
    result = hallucination_rate(_tc(), "zebra elephant giraffe", context="apple banana cherry")
    assert result["hallucination_rate"] == pytest.approx(0.0)


def test_hallucination_rate_missing_context():
    from evals.performance_evals import hallucination_rate
    assert hallucination_rate(_tc(), "some text")["hallucination_rate"] == 0.0


def test_judge_score_trend_stable():
    from evals.performance_evals import judge_score_trend
    scores = [0.8, 0.82, 0.79, 0.81]
    assert judge_score_trend(_tc(), "", judge_scores=scores)["judge_score_trend"] == pytest.approx(1.0, rel=0.1)


def test_judge_score_trend_degrading():
    from evals.performance_evals import judge_score_trend
    scores = [0.9, 0.85, 0.6, 0.5]
    result = judge_score_trend(_tc(), "", judge_scores=scores)["judge_score_trend"]
    assert result < 1.0


def test_judge_score_trend_too_few_points():
    from evals.performance_evals import judge_score_trend
    assert judge_score_trend(_tc(), "", judge_scores=[0.8, 0.7])["judge_score_trend"] == 1.0


def test_user_feedback_score_thumbs():
    from evals.performance_evals import user_feedback_score
    result = user_feedback_score(_tc(), "", thumbs_up=7, thumbs_down=3)
    assert result["user_feedback_score"] == pytest.approx(0.7)


def test_user_feedback_score_explicit_rating():
    from evals.performance_evals import user_feedback_score
    result = user_feedback_score(_tc(), "", explicit_rating=0.95)
    assert result["user_feedback_score"] == pytest.approx(0.95)


def test_user_feedback_score_no_data():
    from evals.performance_evals import user_feedback_score
    assert user_feedback_score(_tc(), "")["user_feedback_score"] == 0.0


# ============================================================================
# RELIABILITY METRICS
# ============================================================================

def test_provider_error_rate_perfect():
    from evals.performance_evals import provider_error_rate
    result = provider_error_rate(_tc(), "", errors=0, total_requests=100)
    assert result["provider_error_rate"] == pytest.approx(1.0)


def test_provider_error_rate_partial():
    from evals.performance_evals import provider_error_rate
    result = provider_error_rate(_tc(), "", errors=10, total_requests=100)
    assert result["provider_error_rate"] == pytest.approx(0.9)


def test_timeout_rate_perfect():
    from evals.performance_evals import timeout_rate
    result = timeout_rate(_tc(), "", timeouts=0, total_requests=50)
    assert result["timeout_rate"] == pytest.approx(1.0)


def test_timeout_rate_partial():
    from evals.performance_evals import timeout_rate
    result = timeout_rate(_tc(), "", timeouts=5, total_requests=100)
    assert result["timeout_rate"] == pytest.approx(0.95)


def test_rate_limit_rate():
    from evals.performance_evals import rate_limit_rate
    result = rate_limit_rate(_tc(), "", rate_limits_hit=20, total_requests=100)
    assert result["rate_limit_rate"] == pytest.approx(0.8)


def test_retry_rate_no_retries():
    from evals.performance_evals import retry_rate
    result = retry_rate(_tc(), "", retries=0, total_requests=100)
    assert result["retry_rate"] == pytest.approx(1.0)


def test_retry_rate_heavy_retries():
    from evals.performance_evals import retry_rate
    result = retry_rate(_tc(), "", retries=200, total_requests=100)
    assert result["retry_rate"] == pytest.approx(0.0)


def test_guardrail_trigger_rate():
    from evals.performance_evals import guardrail_trigger_rate
    result = guardrail_trigger_rate(_tc(), "", guardrail_triggers=15, total_requests=100)
    assert result["guardrail_trigger_rate"] == pytest.approx(0.15)


def test_guardrail_trigger_rate_no_data():
    from evals.performance_evals import guardrail_trigger_rate
    assert guardrail_trigger_rate(_tc(), "")["guardrail_trigger_rate"] == 0.0


# ============================================================================
# AGENT BEHAVIOR METRICS
# ============================================================================

def test_tool_call_error_rate_no_errors():
    from evals.agent_evals import tool_call_error_rate
    trace = _trace(steps=3, tool_errors=0)
    # No tool calls at all → perfect score
    assert tool_call_error_rate(_tc("agent"), "", agent_trace=trace)["tool_call_error_rate"] == 1.0


def test_tool_call_error_rate_with_errors():
    from evals.agent_evals import tool_call_error_rate
    step = AgentStep(
        step_number=1, action="search",
        tool_calls=[
            ToolCall(name="search", actual_result="Error: timeout"),
            ToolCall(name="fetch", actual_result="success"),
        ]
    )
    trace = AgentTrace(agent_id="a1", steps=[step], success=False)
    result = tool_call_error_rate(_tc("agent"), "", agent_trace=trace)["tool_call_error_rate"]
    assert result == pytest.approx(0.5)


def test_llm_calls_per_task_within_budget():
    from evals.agent_evals import llm_calls_per_task
    result = llm_calls_per_task(_tc("agent"), "", llm_call_count=10, max_calls=20)
    assert result["llm_calls_per_task"] == pytest.approx(1.0)


def test_llm_calls_per_task_over_budget():
    from evals.agent_evals import llm_calls_per_task
    result = llm_calls_per_task(_tc("agent"), "", llm_call_count=40, max_calls=20)
    assert result["llm_calls_per_task"] == pytest.approx(0.0)


def test_tokens_per_task_within_budget():
    from evals.agent_evals import tokens_per_task
    result = tokens_per_task(_tc("agent"), "", total_tokens=4000, token_budget=8000)
    assert result["tokens_per_task"] == pytest.approx(1.0)


def test_tokens_per_task_over_budget():
    from evals.agent_evals import tokens_per_task
    result = tokens_per_task(_tc("agent"), "", total_tokens=16000, token_budget=8000)
    assert result["tokens_per_task"] == pytest.approx(0.0)


def test_context_window_utilization_half():
    from evals.agent_evals import context_window_utilization
    result = context_window_utilization(_tc("agent"), "", tokens_used=64000, context_window_size=128000)
    assert result["context_window_utilization"] == pytest.approx(0.5)


def test_context_window_utilization_capped():
    from evals.agent_evals import context_window_utilization
    result = context_window_utilization(_tc("agent"), "", tokens_used=200000, context_window_size=128000)
    assert result["context_window_utilization"] == pytest.approx(1.0)


def test_max_iteration_reached_hit_limit():
    from evals.agent_evals import max_iteration_reached
    trace = _trace(steps=10, success=False)
    result = max_iteration_reached(_tc("agent"), "", agent_trace=trace, max_steps=10)
    assert result["max_iteration_reached"] == pytest.approx(1.0)


def test_max_iteration_reached_completed_cleanly():
    from evals.agent_evals import max_iteration_reached
    trace = _trace(steps=5, success=True)
    result = max_iteration_reached(_tc("agent"), "", agent_trace=trace, max_steps=10)
    assert result["max_iteration_reached"] == pytest.approx(0.0)


# ============================================================================
# RAG @k METRICS
# ============================================================================

def test_precision_at_k_all_relevant():
    from evals.rag_evals import precision_at_k
    tc = _rag_tc(docs=["cat sat mat"])
    chunks = ["the cat sat", "mat floor", "dog barked"]
    result = precision_at_k(tc, "", retrieved_chunks=chunks, k=2)
    assert result["precision_at_k"] == pytest.approx(1.0)


def test_precision_at_k_none_relevant():
    from evals.rag_evals import precision_at_k
    tc = _rag_tc(docs=["cat sat mat"])
    chunks = ["zebra elephant", "lion tiger"]
    result = precision_at_k(tc, "", retrieved_chunks=chunks, k=2)
    assert result["precision_at_k"] == pytest.approx(0.0)


def test_precision_at_k_respects_k():
    from evals.rag_evals import precision_at_k
    tc = _rag_tc(docs=["cat sat mat"])
    # Only first chunk matches; second does not — with k=1 precision should be 1.0
    chunks = ["cat sat", "zebra elephant"]
    result = precision_at_k(tc, "", retrieved_chunks=chunks, k=1)
    assert result["precision_at_k"] == pytest.approx(1.0)


def test_recall_at_k_full_coverage():
    from evals.rag_evals import recall_at_k
    tc = _rag_tc(docs=["cat sat"])
    chunks = ["the cat sat on the mat", "other stuff"]
    result = recall_at_k(tc, "", retrieved_chunks=chunks, k=1)
    assert result["recall_at_k"] == pytest.approx(1.0)


def test_recall_at_k_no_coverage():
    from evals.rag_evals import recall_at_k
    tc = _rag_tc(docs=["cat sat mat"])
    chunks = ["zebra elephant giraffe"]
    result = recall_at_k(tc, "", retrieved_chunks=chunks, k=1)
    assert result["recall_at_k"] == pytest.approx(0.0)


def test_recall_at_k_respects_k():
    from evals.rag_evals import recall_at_k
    tc = _rag_tc(docs=["cat sat mat"])
    # Second chunk has relevant content; k=1 should miss it
    chunks = ["zebra elephant", "cat sat mat"]
    result_k1 = recall_at_k(tc, "", retrieved_chunks=chunks, k=1)
    result_k2 = recall_at_k(tc, "", retrieved_chunks=chunks, k=2)
    assert result_k1["recall_at_k"] < result_k2["recall_at_k"]
