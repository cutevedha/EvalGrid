"""
adapters/llm/openai_adapter.py: Connect EvalGrid to the OpenAI API (GPT models).

Installation
------------
    pip install openai

Authentication
--------------
Pass api_key= explicitly, or set the OPENAI_API_KEY environment variable and
leave api_key=None (the OpenAI SDK picks it up automatically).

Usage
-----
    from adapters.llm.openai_adapter import OpenAIAdapter
    client = OpenAIAdapter(model="gpt-4o-mini")
    response = client.generate_sync("What is 2 + 2?")

Azure OpenAI
------------
This adapter is also compatible with Azure OpenAI endpoints.  Configure the
base URL and API version via the AsyncOpenAI(base_url=..., api_version=...)
constructor if needed.
"""

from typing import List, Dict
import asyncio

from adapters.llm.base import LLMClient


class OpenAIAdapter(LLMClient):
    """
    Async adapter for the OpenAI API (also compatible with Azure OpenAI).

    Requires: pip install openai
    """

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", timeout: int = 30):
        """
        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var if None)
            model: Chat completion model to use
            timeout: Maximum seconds to wait for a response
        """
        try:
            from openai import AsyncOpenAI
            self.client  = AsyncOpenAI(api_key=api_key)
            self.model   = model
            self.timeout = timeout
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> str:
        """
        Generate a text completion for a plain prompt.

        Returns empty string on timeout; error message string on other failures.
        """
        try:
            response = await asyncio.wait_for(
                self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=self.timeout
            )
            return response.choices[0].text
        except asyncio.TimeoutError:
            return ""
        except Exception as e:
            return f"Error: {str(e)}"

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> str:
        """
        Generate a chat completion from a list of {"role": ..., "content": ...} messages.

        Returns empty string on timeout; error message string on other failures.
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=self.timeout
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            return ""
        except Exception as e:
            return f"Error: {str(e)}"

    async def embed(self, text: str, **kwargs) -> List[float]:
        """Generate an embedding vector using text-embedding-3-small"""
        try:
            response = await asyncio.wait_for(
                self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                ),
                timeout=self.timeout
            )
            return response.data[0].embedding
        except asyncio.TimeoutError:
            return []
        except Exception as e:
            return []
