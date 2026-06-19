"""
evalgrid/judge.py: Real LLM-as-judge integration that works out of the box.

The problem this module solves
------------------------------
EvalGrid's LLM judges (correctness, relevance, faithfulness, etc.) call
``client.generate(prompt)`` synchronously. But every adapter exposes the
real ``generate()`` as an async coroutine — calling it from sync code
returns a coroutine object, never a string. The judge code catches that
silently and falls back to keyword heuristics. Result: even users with
OPENAI_API_KEY set were getting fake scores.

What this module provides
-------------------------
1. ``JudgeClient`` — a sync wrapper around any async LLM adapter that
   actually calls ``asyncio.run()`` and returns the string the judge needs.
   It caches identical prompts so the same eval never pays twice.

2. ``auto_detect_judge()`` — reads OPENAI_API_KEY / ANTHROPIC_API_KEY /
   GEMINI_API_KEY and picks a sensible default model.

3. ``set_judge(model_or_client)`` — flip on real LLM judging in one call.

4. ``configure(judge=..., api_key=..., temperature=...)`` — top-level
   one-stop configuration.

Auto-detection priority
-----------------------
   EVALGRID_JUDGE_MODEL env var (explicit override)
   → OPENAI_API_KEY     → gpt-4o-mini
   → ANTHROPIC_API_KEY  → claude-3-haiku
   → GEMINI_API_KEY     → gemini-2.0-flash-exp
   → no env var → return None (heuristics keep working)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("evalgrid.judge")


# ============================================================================
# JUDGE CLIENT — SYNC WRAPPER AROUND ANY ADAPTER
# ============================================================================

class JudgeClient:
    """
    Sync wrapper around an LLM adapter for use as an LLM-as-judge.

    Why this class exists
    ---------------------
    The judge code path is synchronous (a metric is a normal function), but
    every LLM adapter is async. This class bridges that gap: it owns an async
    adapter and exposes a sync ``generate(prompt)`` method that the judge code
    can call directly.

    Features
    --------
    - **Response caching**: identical prompts return the cached string instead
      of hitting the API again. Set ``cache=False`` to disable.
    - **Cost tracking**: optionally attach a CostTracker that records every
      real LLM call (not heuristic fallbacks).
    - **Error safety**: API errors, timeouts, and asyncio issues all become
      empty strings — the calling judge then falls back to heuristics.

    Example
    -------
        from evalgrid import JudgeClient
        from adapters.llm.openai_adapter import OpenAIAdapter

        client = JudgeClient(OpenAIAdapter(model="gpt-4o-mini"), temperature=0)
        client.generate("Score this on a scale of 1-5: ...")
    """

    def __init__(
        self,
        adapter: Any,
        temperature: float = 0.0,
        max_tokens: int = 500,
        cache: bool = True,
        cost_tracker: Optional[Any] = None,
    ) -> None:
        """
        Args:
            adapter:      Any LLM adapter with an async ``generate()`` method.
            temperature:  Judge temperature. Default 0.0 for deterministic scoring.
            max_tokens:   Max response length. 500 is plenty for "REASONING: ... SCORE: X".
            cache:        Cache identical prompts for free re-evaluation.
            cost_tracker: Optional CostTracker that gets called per real LLM call.
        """
        self.adapter = adapter
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_tracker = cost_tracker
        self._cache: Optional[Dict[str, str]] = {} if cache else None
        self._real_calls = 0
        self._cache_hits = 0
        self._errors = 0
        # Lock for concurrent stats updates from many worker threads.
        self._stats_lock = threading.Lock()

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Sync LLM call. Returns the model's response as a string.

        On any error (network, timeout, asyncio issue, model refusal) returns "",
        which causes the calling judge to fall back to its heuristic scorer.

        Accepts ``**kwargs`` so existing judge code passing keyword arguments
        continues to work — they are merged on top of our defaults.
        """
        # Cache lookup — identical prompt = identical response
        if self._cache is not None:
            key = self._cache_key(prompt)
            cached = self._cache.get(key)
            if cached is not None:
                with self._stats_lock:
                    self._cache_hits += 1
                return cached

        try:
            response = self.adapter.generate_sync(
                prompt,
                temperature=kwargs.pop("temperature", self.temperature),
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                **kwargs,
            )
            with self._stats_lock:
                self._real_calls += 1
            if self.cost_tracker is not None:
                try:
                    self.cost_tracker.record(
                        "llm_judge",
                        input_text=prompt,
                        output_text=response,
                    )
                except Exception:
                    pass  # Cost tracking is best-effort
            if self._cache is not None:
                self._cache[key] = response or ""
            return response or ""
        except Exception as exc:
            with self._stats_lock:
                self._errors += 1
            logger.debug("JudgeClient.generate() failed: %s", exc)
            return ""

    @property
    def model(self) -> str:
        """Best-effort model identifier for reports and logs."""
        return getattr(self.adapter, "model", self.adapter.__class__.__name__)

    def stats(self) -> Dict[str, int]:
        """Return call statistics for the judge's lifetime."""
        return {
            "real_calls": self._real_calls,
            "cache_hits": self._cache_hits,
            "errors":     self._errors,
            "cached_items": len(self._cache) if self._cache is not None else 0,
        }

    def clear_cache(self) -> None:
        """Drop all cached responses. Subsequent identical prompts will re-call the API."""
        if self._cache is not None:
            self._cache.clear()

    @staticmethod
    def _cache_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# ============================================================================
# MODEL NAME → ADAPTER ROUTING
# ============================================================================

# Default model per provider — cheap, fast, capable enough for judging.
_DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "gemini":    "gemini-2.0-flash-exp",
}

_AUTO_DETECT_PRIORITY = [
    ("OPENAI_API_KEY",    "openai"),
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("GEMINI_API_KEY",    "gemini"),
    ("GOOGLE_API_KEY",    "gemini"),
]


def _provider_for_model(model: str) -> str:
    """Map a model name to a provider. Defaults to openai for unknown prefixes."""
    model_lower = model.lower()
    if model_lower.startswith(("gpt", "o1", "o3", "text-")):
        return "openai"
    if model_lower.startswith("claude"):
        return "anthropic"
    if model_lower.startswith("gemini"):
        return "gemini"
    if any(model_lower.startswith(p) for p in ("llama", "mistral", "gemma", "qwen", "phi")):
        return "ollama"
    return "openai"


def _build_adapter(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """Construct the right LLM adapter for a given model identifier."""
    provider = _provider_for_model(model)
    if provider == "openai":
        from adapters.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key=api_key, model=model, base_url=base_url)
    if provider == "anthropic":
        from adapters.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key=api_key, model=model, base_url=base_url)
    if provider == "gemini":
        from adapters.llm.gemini_adapter import GeminiAdapter
        return GeminiAdapter(api_key=api_key, model=model)
    # Ollama — typically a local server; api_key optional
    from adapters.llm.ollama_adapter import OllamaAdapter
    return OllamaAdapter(
        base_url=base_url or "http://localhost:11434",
        model=model,
        api_key=api_key,
    )


# ============================================================================
# AUTO DETECTION
# ============================================================================

def auto_detect_judge(
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> Optional[JudgeClient]:
    """
    Pick the best available judge based on environment variables.

    Returns a configured JudgeClient or None when no provider is available
    (in which case the framework keeps using its heuristic fallbacks).

    Honors ``EVALGRID_JUDGE_MODEL`` for an explicit override:
        export EVALGRID_JUDGE_MODEL="claude-3-5-sonnet-20241022"
    """
    explicit_model = os.getenv("EVALGRID_JUDGE_MODEL")
    if explicit_model:
        try:
            adapter = _build_adapter(explicit_model, api_key=api_key)
            return JudgeClient(adapter, temperature=temperature)
        except Exception as exc:
            logger.debug("auto_detect_judge: explicit model %r failed: %s", explicit_model, exc)
            return None

    for env_var, provider in _AUTO_DETECT_PRIORITY:
        if not os.getenv(env_var):
            continue
        model = _DEFAULT_MODELS[provider]
        try:
            adapter = _build_adapter(model, api_key=api_key or os.getenv(env_var))
            return JudgeClient(adapter, temperature=temperature)
        except Exception as exc:
            logger.debug("auto_detect_judge: %s failed: %s", provider, exc)
            continue
    return None


# ============================================================================
# GLOBAL STATE — set_judge / get_judge / configure
# ============================================================================

# Module-level marker so we know whether auto-detection has already run.
_auto_detected = False


def set_judge(judge: Union[str, JudgeClient, Any, None]) -> Optional[JudgeClient]:
    """
    Configure the global judge used by every LLM-based metric.

    Accepts three forms:

    1. **Model name string**
           set_judge("gpt-4o-mini")
       Builds the right adapter from the env var and wraps it in JudgeClient.

    2. **A JudgeClient instance**
           set_judge(JudgeClient(my_adapter, temperature=0))

    3. **A raw LLM adapter**
           set_judge(MyCustomAdapter())
       Auto-wrapped in a JudgeClient.

    Pass ``None`` to disable the judge and force heuristic scoring.
    """
    from evals.llm_judge import set_llm_client

    if judge is None:
        set_llm_client(None)
        return None

    if isinstance(judge, JudgeClient):
        client = judge
    elif isinstance(judge, str):
        adapter = _build_adapter(judge)
        client = JudgeClient(adapter)
    else:
        # Assume it's already an LLM adapter; wrap it.
        client = JudgeClient(judge)

    set_llm_client(client)
    return client


def get_judge() -> Optional[JudgeClient]:
    """Return the currently active judge client, or None when heuristics are in use."""
    from evals.llm_judge import get_llm_client
    client = get_llm_client()
    return client if isinstance(client, JudgeClient) else None


def configure(
    judge: Optional[Union[str, JudgeClient, Any]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
    cache: bool = True,
) -> Optional[JudgeClient]:
    """
    Top-level one-stop configuration for the EvalGrid judge.

    Examples
    --------
        from evalgrid import configure

        # Use GPT-4o-mini with env var OPENAI_API_KEY
        configure(judge="gpt-4o-mini")

        # Use Claude with explicit key
        configure(judge="claude-3-5-sonnet-20241022", api_key="sk-ant-...")

        # Route through Azure OpenAI
        configure(judge="gpt-4o", api_key="...", base_url="https://my.azure.com")

        # Disable the judge and force heuristic mode
        configure(judge=None)
    """
    if judge is None:
        from evals.llm_judge import set_llm_client
        set_llm_client(None)
        return None

    if isinstance(judge, JudgeClient):
        from evals.llm_judge import set_llm_client
        set_llm_client(judge)
        return judge

    if isinstance(judge, str):
        adapter = _build_adapter(judge, api_key=api_key, base_url=base_url)
    else:
        adapter = judge

    client = JudgeClient(
        adapter,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
    )
    from evals.llm_judge import set_llm_client
    set_llm_client(client)
    return client


# ============================================================================
# LAZY AUTO-DETECTION (called by evaluate() on first use)
# ============================================================================

def ensure_judge_configured() -> Optional[JudgeClient]:
    """
    Ensure a judge is available, auto-detecting from env vars if needed.

    Called by ``evaluate()`` on the first run that uses an LLM-judge metric.
    Subsequent calls are no-ops, so we never re-run detection unnecessarily.

    Returns the active JudgeClient or None if no provider is available.
    """
    global _auto_detected
    from evals.llm_judge import get_llm_client

    existing = get_llm_client()
    if existing is not None:
        return existing if isinstance(existing, JudgeClient) else None

    if _auto_detected:
        # Already attempted; no provider was available, don't waste time again.
        return None

    _auto_detected = True

    if os.getenv("EVALGRID_DISABLE_AUTO_JUDGE"):
        return None

    judge_client = auto_detect_judge()
    if judge_client is not None:
        from evals.llm_judge import set_llm_client
        set_llm_client(judge_client)
        logger.info("EvalGrid: auto-detected judge model %r", judge_client.model)
    return judge_client


def reset_auto_detection() -> None:
    """Reset the auto-detection flag — primarily useful in test setup."""
    global _auto_detected
    _auto_detected = False
