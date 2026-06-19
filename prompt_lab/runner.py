"""
prompt_lab/runner.py: Send a prompt to ChatGPT, Gemini, and Copilot in parallel.

Each LLM is driven by the existing EvalGrid adapter layer.  Copilot is accessed
via GitHub Models (OpenAI-compatible endpoint) — same GPT-4o model that powers
Microsoft Copilot; you just need a GitHub personal access token.

Environment variables (add to .env):
    OPENAI_API_KEY      – for ChatGPT
    GEMINI_API_KEY      – for Gemini
    GITHUB_TOKEN        – for Copilot (GitHub Models)

Any key that is missing causes that LLM to be skipped gracefully with a clear
"not configured" message rather than crashing.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


LLM_NAMES = ["ChatGPT", "Gemini", "Copilot"]


@dataclass
class LLMResult:
    llm: str                  # "ChatGPT" | "Gemini" | "Copilot"
    response: str             # raw text response
    latency_ms: int           # wall-clock time in milliseconds
    error: Optional[str] = None  # set when the call failed / was skipped


# ---------------------------------------------------------------------------
# Adapter factories
# ---------------------------------------------------------------------------

def _build_chatgpt():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None, "OPENAI_API_KEY not set in .env"
    from adapters.llm.openai_adapter import OpenAIAdapter
    return OpenAIAdapter(api_key=key, model="gpt-4o-mini", timeout=60), None


def _build_gemini():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None, "GEMINI_API_KEY not set in .env"
    try:
        from adapters.llm.gemini_adapter import GeminiAdapter
        return GeminiAdapter(api_key=key, model="gemini-1.5-flash", timeout=60), None
    except ImportError:
        return None, "google-generativeai not installed. Run: pip install google-generativeai"


def _build_copilot():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return None, "GITHUB_TOKEN not set in .env  (GitHub personal access token needed for Copilot)"
    from adapters.llm.openai_adapter import OpenAIAdapter
    return OpenAIAdapter(
        api_key=token,
        model="gpt-4o",
        base_url="https://models.inference.ai.azure.com",
        timeout=60,
    ), None


_FACTORIES = {
    "ChatGPT": _build_chatgpt,
    "Gemini":  _build_gemini,
    "Copilot": _build_copilot,
}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def _call_one(name: str, prompt_text: str) -> LLMResult:
    factory = _FACTORIES[name]
    adapter, err = factory()
    if err:
        return LLMResult(llm=name, response="", latency_ms=0, error=err)

    start = time.monotonic()
    try:
        response = await adapter.generate(prompt_text, max_tokens=1500)
    except Exception as exc:
        response = ""
        err = str(exc)
    elapsed = int((time.monotonic() - start) * 1000)
    return LLMResult(llm=name, response=response, latency_ms=elapsed, error=err)


async def run_all(prompt_text: str, llms: Optional[List[str]] = None) -> List[LLMResult]:
    """
    Send prompt_text to the requested LLMs in parallel.

    Args:
        prompt_text: The prompt string to test.
        llms:        Which LLMs to use. Defaults to all three.

    Returns:
        List of LLMResult objects in the same order as llms.
    """
    targets = llms or LLM_NAMES
    tasks = [_call_one(name, prompt_text) for name in targets]
    return list(await asyncio.gather(*tasks))


def run_all_sync(prompt_text: str, llms: Optional[List[str]] = None) -> List[LLMResult]:
    """Synchronous wrapper for run_all()."""
    return asyncio.run(run_all(prompt_text, llms))
