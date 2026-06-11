"""
adapters/llm/anthropic_adapter.py: Connect EvalGrid to Anthropic's Claude models.

Installation
------------
    pip install anthropic

Authentication
--------------
Pass api_key= explicitly, or set the ANTHROPIC_API_KEY environment variable and
leave api_key=None (the Anthropic SDK picks it up automatically).

Usage
-----
    from adapters.llm.anthropic_adapter import AnthropicAdapter
    client = AnthropicAdapter(model="claude-sonnet-4-6")
    response = client.generate_sync("What is 2 + 2?")

Note on embeddings
------------------
Anthropic does not currently expose a public embedding API endpoint.
embed() returns an empty list; use a different embedder (e.g. OpenAIEmbedder
from evals/semantic.py) if you need semantic similarity metrics.
"""

from typing import List, Dict
import asyncio

from adapters.llm.base import LLMClient


class AnthropicAdapter(LLMClient):
    """
    Async adapter for the Anthropic Messages API (Claude models).

    Requires: pip install anthropic
    """

    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229", timeout: int = 30):
        """
        Args:
            api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var if None)
            model: Claude model ID to use
            timeout: Maximum seconds to wait for a response
        """
        try:
            from anthropic import AsyncAnthropic
            self.client  = AsyncAnthropic(api_key=api_key)
            self.model   = model
            self.timeout = timeout
        except ImportError:
            raise ImportError("anthropic not installed. Install with: pip install anthropic")

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> str:
        """Wrap a plain prompt in a single user message and call the Messages API"""
        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                ),
                timeout=self.timeout
            )
            return response.content[0].text
        except asyncio.TimeoutError:
            return ""
        except Exception as e:
            return f"Error: {str(e)}"

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> str:
        """Send a multi-turn conversation to the Messages API"""
        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=temperature,
                ),
                timeout=self.timeout
            )
            return response.content[0].text
        except asyncio.TimeoutError:
            return ""
        except Exception as e:
            return f"Error: {str(e)}"

    async def embed(self, text: str, **kwargs) -> List[float]:
        """Anthropic does not expose a public embedding endpoint; returns empty list"""
        return []
