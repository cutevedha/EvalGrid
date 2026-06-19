"""
adapters/llm/gemini_adapter.py: Connect EvalGrid to Google's Gemini models.

Installation
------------
    pip install google-generativeai

Authentication
--------------
Pass api_key= explicitly, or set the GEMINI_API_KEY environment variable.

Usage
-----
    from adapters.llm.gemini_adapter import GeminiAdapter
    client = GeminiAdapter(model="gemini-1.5-flash")
    response = client.generate_sync("Explain test-driven development in 3 bullet points.")
"""

from typing import List, Dict, Optional
import asyncio
import os

from adapters.llm.base import LLMClient


class GeminiAdapter(LLMClient):
    """
    Async adapter for Google Gemini via the google-generativeai SDK.

    Requires: pip install google-generativeai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        timeout: int = 60,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY in your .env file "
                "or pass api_key= directly."
            )
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._genai = genai
            self.model_name = model
            self.timeout = timeout
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )

    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            loop = asyncio.get_event_loop()
            model = self._genai.GenerativeModel(self.model_name)
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: model.generate_content(prompt)),
                timeout=self.timeout,
            )
            return response.text
        except asyncio.TimeoutError:
            return ""
        except Exception as e:
            return f"Error: {e}"

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Flatten chat messages into a single prompt for Gemini
        combined = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        return await self.generate(combined, **kwargs)

    async def embed(self, text: str, **kwargs) -> List[float]:
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._genai.embed_content(
                        model="models/embedding-001", content=text
                    ),
                ),
                timeout=self.timeout,
            )
            return result["embedding"]
        except Exception:
            return []
