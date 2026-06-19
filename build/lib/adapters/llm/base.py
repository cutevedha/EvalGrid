"""
adapters/llm/base.py: Shared interface that every LLM adapter must implement.

Why a base class?
-----------------
EvalGrid supports multiple AI providers (Anthropic, OpenAI, Ollama, …).  Having
all adapters implement the same interface means the rest of the framework never
needs to know *which* provider is in use: it just calls generate(), chat(), or
embed() and gets back a string or vector.

Design decisions
----------------
- All methods are **async** (non-blocking) by default because LLM API calls can
  take several seconds and we often want to run many in parallel.
- **Synchronous wrappers** (generate_sync, chat_sync, embed_sync) are provided
  for callers that cannot use async/await: they simply run the async version in
  a new event loop via asyncio.run().

Implementing a new adapter
--------------------------
1. Create a new file under adapters/llm/.
2. Subclass LLMClient.
3. Override generate(), chat(), and embed().
4. Return empty string / empty list on errors rather than raising, so one bad
   LLM call never aborts an entire evaluation batch.
"""

from typing import Any, Awaitable, List, Dict, TypeVar
import asyncio

_T = TypeVar("_T")


# ============================================================================
# BASE LLM CLIENT INTERFACE
# ============================================================================

class LLMClient:
    """
    Abstract base class for all LLM adapters.

    Every adapter must implement async versions of generate(), chat(), and embed().
    Synchronous wrappers (generate_sync, etc.) are provided here automatically via
    asyncio.run() so callers that cannot use async/await still work.
    """

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a completion for a plain text prompt"""
        raise NotImplementedError

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat completion from a list of role/content messages"""
        raise NotImplementedError

    async def embed(self, text: str, **kwargs) -> List[float]:
        """Return a vector embedding for the given text"""
        raise NotImplementedError

    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper around generate()"""
        return asyncio.run(self.generate(prompt, **kwargs))

    def chat_sync(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Synchronous wrapper around chat()"""
        return asyncio.run(self.chat(messages, **kwargs))

    def embed_sync(self, text: str, **kwargs) -> List[float]:
        """Synchronous wrapper around embed()"""
        return asyncio.run(self.embed(text, **kwargs))

    @staticmethod
    async def _with_timeout(coro: Awaitable[_T], timeout: float) -> _T:
        """
        Await ``coro`` with a hard deadline and re-raise ``asyncio.TimeoutError`` on breach.

        Subclasses call this instead of repeating ``asyncio.wait_for`` inline,
        keeping timeout handling in one place.
        """
        return await asyncio.wait_for(coro, timeout=timeout)
