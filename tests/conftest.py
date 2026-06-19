"""
tests/conftest.py: Pytest-wide setup for EvalGrid.

These hooks ensure the test suite never hits a real LLM API:

  • EVALGRID_DISABLE_AUTO_JUDGE=1 prevents ``evalgrid.judge.ensure_judge_configured()``
    from auto-detecting OPENAI_API_KEY / ANTHROPIC_API_KEY during tests.

  • The global judge client is cleared before each test so a previous test that
    explicitly set a judge can't leak its mock or real client into the next.

Individual tests that DO want to exercise the LLM-judge code path should set
a JudgeClient explicitly via ``set_judge(mock_client)``.
"""

import os

import pytest


def pytest_configure(config):
    """Disable auto-detection of real LLM judges for the whole suite."""
    os.environ.setdefault("EVALGRID_DISABLE_AUTO_JUDGE", "1")


@pytest.fixture(autouse=True)
def _reset_global_judge():
    """Clear any judge that a prior test may have left set."""
    try:
        from evals.llm_judge import set_llm_client
        from evalgrid.judge import reset_auto_detection
        set_llm_client(None)
        reset_auto_detection()
    except ImportError:
        pass
    yield
    try:
        from evals.llm_judge import set_llm_client
        from evalgrid.judge import reset_auto_detection
        set_llm_client(None)
        reset_auto_detection()
    except ImportError:
        pass
