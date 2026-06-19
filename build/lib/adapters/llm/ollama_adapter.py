"""
adapters/llm/ollama_adapter.py: Connect EvalGrid to open-source models via Ollama.

Ollama lets you run large language models locally on your own machine: no API
key, no usage fees, and full privacy.

Prerequisites
-------------
1. Install Ollama:  https://ollama.ai
2. Pull a model:    ollama pull llama2
3. Install httpx:   pip install httpx

Authentication
--------------
Local Ollama typically requires no API key. When running a remote or
protected Ollama instance (e.g. behind an auth-proxy), pass api_key= and it
will be sent as a Bearer token in the Authorization header:

    OllamaAdapter(base_url="https://ollama.corp.example.com", api_key="secret")

Usage
-----
    from adapters.llm.ollama_adapter import OllamaAdapter
    client = OllamaAdapter(model="llama2")
    response = client.generate_sync("What is 2 + 2?")

The adapter connects to a locally running Ollama server (default port 11434).
To use a remote server, pass a different base_url.
"""

from typing import List, Dict, Optional
import asyncio

from adapters.llm.base import LLMClient


class OllamaAdapter(LLMClient):
    """
    Async adapter for Ollama: a local model server that runs open-source LLMs.

    Supports an optional api_key passed as a Bearer token — useful when Ollama
    is deployed behind an authentication proxy rather than on localhost.

    Requires: pip install httpx  +  a running Ollama instance
    Docs: https://ollama.ai
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: int = 30,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            base_url: Ollama server URL (default: local).
            model:    Model name to use (must be pulled via `ollama pull <model>`).
            timeout:  Maximum seconds to wait for a response.
            api_key:  Optional bearer token for protected Ollama deployments.
                      When None no Authorization header is sent (standard for local installs).
        """
        try:
            import httpx
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            self.client   = httpx.AsyncClient(base_url=base_url, headers=headers)
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
