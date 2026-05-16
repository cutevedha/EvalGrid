# LLM Adapter Base (legacy sync stub)
# Kept for backwards compatibility — new code should use openai_adapter.py which provides
# the full async LLMClient base class with OpenAI, Anthropic, Ollama, and Mock implementations

class LLMClient:
    """Minimal synchronous stub for the LLM client interface (legacy)"""

    def generate(self, prompt: str) -> str:
        """
        Generate a completion for a plain text prompt.

        Args:
            prompt: The input prompt string

        Returns:
            Generated text response
        """
        raise NotImplementedError
