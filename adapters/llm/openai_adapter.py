# LLM Adapters - Async-first clients for OpenAI, Anthropic, Ollama, and Mock
# All adapters share the LLMClient base interface so they are interchangeable

from typing import List, Optional, Dict, Any
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


# ============================================================================
# OPENAI ADAPTER
# ============================================================================

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


# ============================================================================
# ANTHROPIC ADAPTER
# ============================================================================

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


# ============================================================================
# OLLAMA ADAPTER (LOCAL MODELS)
# ============================================================================

class OllamaAdapter(LLMClient):
    """
    Async adapter for Ollama — a local model server that runs open-source LLMs.

    Requires: pip install httpx  +  a running Ollama instance
    Docs: https://ollama.ai
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2", timeout: int = 30):
        """
        Args:
            base_url: Ollama server URL (default: local)
            model: Model name to use (must be pulled via `ollama pull <model>`)
            timeout: Maximum seconds to wait for a response
        """
        try:
            import httpx
            self.client   = httpx.AsyncClient(base_url=base_url)
            self.base_url = base_url
            self.model    = model
            self.timeout  = timeout
        except ImportError:
            raise ImportError("httpx not installed. Install with: pip install httpx")

    async def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """POST to /api/generate and return the response text"""
        try:
            response = await asyncio.wait_for(
                self.client.post(
                    "/api/generate",
                    json={
                        "model":       self.model,
                        "prompt":      prompt,
                        "temperature": temperature,
                        "stream":      False,  # Disable streaming to get a single response
                    },
                ),
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except asyncio.TimeoutError:
            return ""
        except Exception as e:
            return f"Error: {str(e)}"

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, **kwargs) -> str:
        """Convert chat messages to a plain prompt and call generate()"""
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return await self.generate(prompt, temperature)

    async def embed(self, text: str, **kwargs) -> List[float]:
        """Embedding not implemented for Ollama in this adapter; returns empty list"""
        return []


# ============================================================================
# MOCK ADAPTER (TESTING)
# ============================================================================

class MockLLMAdapter(LLMClient):
    """
    In-memory mock adapter for unit tests.

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
