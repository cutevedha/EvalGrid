# LLM Adapter Base - Async-first client interface shared by every adapter
# OpenAIAdapter, AnthropicAdapter, OllamaAdapter and MockLLMAdapter all subclass
# LLMClient so they are interchangeable. Synchronous wrappers are provided here via
# asyncio.run() so callers that cannot use async/await still work.

from typing import List, Dict
import asyncio


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
