# Anthropic Adapter - Async client for the Anthropic Messages API (Claude models)
# Requires: pip install anthropic

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
