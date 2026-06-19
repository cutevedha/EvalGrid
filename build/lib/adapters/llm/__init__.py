# LLM adapters package
# Each adapter lives in its own module; they are re-exported here so callers can do
#   from adapters.llm import OpenAIAdapter, MockLLMAdapter
# instead of importing each module path individually.

from adapters.llm.base import LLMClient
from adapters.llm.openai_adapter import OpenAIAdapter
from adapters.llm.anthropic_adapter import AnthropicAdapter
from adapters.llm.ollama_adapter import OllamaAdapter
from adapters.llm.mock_target_adapter import MockLLMAdapter

__all__ = [
    "LLMClient",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
    "MockLLMAdapter",
]
