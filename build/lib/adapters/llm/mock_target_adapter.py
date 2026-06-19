"""
adapters/llm/mock_target_adapter.py: Fake LLM for testing without API keys.

The MockLLMAdapter stands in for a real LLM during unit tests and CI runs.
It returns fixed, deterministic responses instantly: no network calls, no
rate limits, no costs.

When to use it
--------------
- Automated tests that should not depend on external services.
- Demonstrating EvalGrid features (eval-grid auto --target mock).
- Developing new metrics: you need an output to evaluate, not a real LLM answer.

The mock tracks call_count so your tests can assert that the adapter was invoked
the expected number of times.
"""

from typing import List, Dict

from adapters.llm.base import LLMClient


class MockLLMAdapter(LLMClient):
    """
    In-memory mock adapter for unit tests and the CLI `--target mock`.

    Returns deterministic placeholder responses without making any network calls.
    Tracks call_count so tests can assert how many times the adapter was invoked.
    """

    def __init__(self):
        self.call_count = 0  # Incremented on every generate() or chat() call

    async def generate(self, prompt: str, **kwargs) -> str:
        """Return a truncated echo of the prompt as a fake completion"""
        self.call_count += 1
        return f"Mock response to: {prompt[:50]}..."

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Return a fixed mock chat response"""
        self.call_count += 1
        return "Mock chat response"

    async def embed(self, text: str, **kwargs) -> List[float]:
        """Return a fixed-length zero-ish embedding vector (384 dims)"""
        return [0.1] * 384
