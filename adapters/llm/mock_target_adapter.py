# Mock Target Adapter - In-memory LLM stub for tests and the `auto` mock target
# Returns deterministic placeholder responses without any network calls, so the
# autonomous agent and the test suite can run with no API keys or external services.

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
