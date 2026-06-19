"""
evalgrid: The user-friendly evaluation framework for LLMs and AI systems.

Quick start
-----------
    from evalgrid import evaluate, TestCase

    results = evaluate(
        cases=[
            TestCase(input="What is AI?", expected_output="AI stands for artificial intelligence."),
        ],
        metrics=["correctness", "relevance", "toxicity"],
    )
    print(results.summary())
    results.to_html("report.html")

Even shorter — pass raw dicts:
    results = evaluate(
        cases=[{"input": "What is AI?", "expected_output": "AI is..."}],
        metrics="generation",   # preset
    )

Pytest:
    from evalgrid import assert_test

    def test_my_chatbot():
        assert_test(
            input="Hello",
            output=my_chatbot("Hello"),
            metrics=["correctness", "relevance"],
            threshold=0.7,
        )
"""

from core.schemas import (
    AgentEvalResult,
    AgentTestCase,
    AgentTrace,
    EvalResult,
    MultiTurnTestCase,
    RAGEvalResult,
    RAGTestCase,
    TestCase,
)


# ── Eager metric registration ─────────────────────────────────────────────
# Importing `evalgrid` should give the user a fully-stocked MetricRegistry
# without forcing them to know about internal module paths.
def _register_all_metrics() -> None:
    """Import every metric module so its @register_metric decorators run."""
    import importlib
    metric_modules = [
        "evals.deterministic",
        "evals.semantic",
        "evals.llm_judge",
        "evals.reference_judge",
        "evals.summarization_evals",
        "evals.structured_evals",
        "evals.performance_evals",
        "evals.agent_evals",
        "evals.rag_evals",
        "evals.bias_fairness_evals",
        "evals.robustness_evals",
        "evals.embedded_ai_evals",
        "evals.multiagent_evals",
        "evals.custom_metrics",
        "guards.toxicity",
        "guards.hallucination",
        "guards.pii",
        "guards.prompt_injection",
    ]
    for module_name in metric_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            # A missing optional dep should not stop the user from using the rest
            pass


_register_all_metrics()


from evalgrid.evaluate import EvalRun, a_evaluate, evaluate, quick_eval
from evalgrid.presets import MetricSet
from evalgrid.pytest_plugin import assert_each, assert_test
from evalgrid.cost import CostTracker
from evalgrid.cache import ScoreCache
from evalgrid.judge import (
    JudgeClient,
    auto_detect_judge,
    configure,
    get_judge,
    set_judge,
)

__all__ = [
    "TestCase",
    "AgentTestCase",
    "RAGTestCase",
    "MultiTurnTestCase",
    "EvalResult",
    "AgentEvalResult",
    "RAGEvalResult",
    "AgentTrace",
    "evaluate",
    "a_evaluate",
    "quick_eval",
    "EvalRun",
    "MetricSet",
    "assert_test",
    "assert_each",
    "CostTracker",
    "ScoreCache",
    "JudgeClient",
    "configure",
    "set_judge",
    "get_judge",
    "auto_detect_judge",
]

__version__ = "1.0.0"
