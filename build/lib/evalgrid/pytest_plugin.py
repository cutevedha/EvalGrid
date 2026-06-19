"""
evalgrid/pytest_plugin.py: Pytest helpers for asserting LLM behaviour.

Drop-in assertions you can use inside any pytest test:

    from evalgrid import assert_test

    def test_my_chatbot():
        assert_test(
            input="What is gravity?",
            output=my_chatbot("What is gravity?"),
            expected="Gravity is a fundamental force...",
            metrics=["llm_judge_correctness", "llm_judge_relevance"],
            threshold=0.7,
        )

For batch testing across many cases:

    from evalgrid import assert_each

    @pytest.mark.parametrize("case", load_dataset("qa_tests.xlsx"))
    def test_each_case(case):
        output = my_chatbot(case.input)
        assert_each(case, output, metrics="generation", threshold=0.7)
"""

from typing import List, Optional, Union

from core.schemas import TestCase
from evalgrid.evaluate import evaluate


def assert_test(
    input: str,
    output: str,
    expected: Optional[str] = None,
    context: Optional[str] = None,
    metrics: Union[str, List] = "generation",
    threshold: float = 0.5,
    test_id: Optional[str] = None,
) -> None:
    """
    Assert a single (input, output) pair passes the chosen metrics.

    Raises AssertionError with a friendly message that lists every failing metric
    and its score, so pytest output tells you exactly what went wrong.
    """
    case = TestCase(
        id=test_id or "pytest_case",
        project="pytest",
        capability="generation",
        input=input,
        expected_output=expected,
        context=context,
    )
    run = evaluate(
        cases=[(case, output)],
        metrics=metrics,
        threshold=threshold,
        progress=False,
        quiet=True,
    )
    result = run.results[0]

    if not result.passed:
        failed_details = "\n".join(
            f"  ✗ {m}: {result.scores.get(m, 'N/A'):.3f} (threshold: {threshold})"
            for m in result.failed_metrics
        )
        raise AssertionError(
            f"\nEvalGrid assertion failed for test '{case.id}':\n"
            f"  input:  {input[:100]}...\n"
            f"  output: {output[:100]}...\n"
            f"Failed metrics:\n{failed_details}"
        )


def assert_each(
    case: TestCase,
    output: str,
    metrics: Union[str, List] = "generation",
    threshold: float = 0.5,
) -> None:
    """
    Like ``assert_test`` but takes a pre-built TestCase — handy with parametrize.
    """
    run = evaluate(
        cases=[(case, output)],
        metrics=metrics,
        threshold=threshold,
        progress=False,
        quiet=True,
    )
    result = run.results[0]
    if not result.passed:
        failed_details = "\n".join(
            f"  ✗ {m}: {result.scores.get(m, 'N/A'):.3f} (threshold: {threshold})"
            for m in result.failed_metrics
        )
        raise AssertionError(
            f"\nEvalGrid assertion failed for test '{case.id}':\n"
            f"  input:  {case.input[:100]}...\n"
            f"  output: {output[:100]}...\n"
            f"Failed metrics:\n{failed_details}"
        )
