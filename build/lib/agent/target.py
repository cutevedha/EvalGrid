# Eval Target - Uniform async wrapper around any system-under-test (SUT)
# The autonomous EvalAgent only ever talks to a target through this interface, so the
# same agent can drive an LLM client, an arbitrary callable, or a pre-computed output map.

from __future__ import annotations

from typing import Awaitable, Callable, Dict, Optional, Union
import asyncio
import inspect

from core.schemas import TestCase


# A target function may be sync or async and is always called with (input, context).
TargetFn = Callable[[str, Optional[str]], Union[str, Awaitable[str]]]


# ============================================================================
# EVAL TARGET
# ============================================================================

class EvalTarget:
    """
    Uniform async wrapper around the system being evaluated.

    The agent calls ``await target.run(test_case)`` and always receives a plain
    output string back, regardless of what the underlying system actually is.
    Construct one with the appropriate factory method:

        EvalTarget.from_llm(client)              # any adapters.llm LLMClient
        EvalTarget.from_callable(my_fn)          # sync or async fn(input, context) -> str
        EvalTarget.from_outputs({"t1": "..."})   # offline, pre-computed outputs
    """

    def __init__(self, fn: TargetFn, name: str = "target"):
        """
        Args:
            fn: Callable taking (input, context) and returning an output string
                (or an awaitable that resolves to one).
            name: Human-readable label used in reports.
        """
        self._fn = fn
        self.name = name

    # ------------------------------------------------------------------
    # FACTORY METHODS
    # ------------------------------------------------------------------

    @classmethod
    def from_llm(cls, client, name: str = None, **gen_kwargs) -> "EvalTarget":
        """
        Wrap an LLM adapter (OpenAI/Anthropic/Ollama/Mock) as a target.

        The test case input and context are folded into a single prompt before
        being passed to ``client.generate``.

        Args:
            client: Any adapters.llm LLMClient instance (must implement async generate())
            name: Optional label (defaults to the client class name)
            **gen_kwargs: Extra keyword args forwarded to generate() (temperature, etc.)
        """
        async def _call(input_text: str, context: Optional[str]) -> str:
            prompt = input_text if not context else f"Context:\n{context}\n\nTask:\n{input_text}"
            return await client.generate(prompt, **gen_kwargs)

        return cls(_call, name=name or client.__class__.__name__)

    @classmethod
    def from_callable(cls, fn: TargetFn, name: str = "callable") -> "EvalTarget":
        """
        Wrap an arbitrary sync or async callable ``fn(input, context) -> str``.

        Callables that only accept a single positional argument are also supported  - 
        the context is dropped automatically.
        """
        sig_params = _count_positional_params(fn)

        async def _call(input_text: str, context: Optional[str]) -> str:
            if sig_params <= 1:
                result = fn(input_text)
            else:
                result = fn(input_text, context)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, str) else str(result)

        return cls(_call, name=name)

    @classmethod
    def from_outputs(cls, outputs: Dict[str, str], name: str = "offline") -> "EvalTarget":
        """
        Wrap a pre-computed output map for fully offline evaluation.

        Lookup order for each test case: by test-case id, then by raw input text.
        A missing entry yields an empty string (which most safety metrics treat as safe).
        """
        async def _call(input_text: str, context: Optional[str]) -> str:
            return outputs.get(_call.current_id, outputs.get(input_text, ""))

        _call.current_id = ""  # set by run() right before invocation
        target = cls(_call, name=name)
        target._is_offline = True
        return target

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    async def run(self, test_case: TestCase) -> str:
        """
        Produce an output for a single test case.

        Errors raised by the underlying system are swallowed and returned as an
        ``"Error: ..."`` string so a single bad call never aborts a whole run.
        """
        # Offline targets need the id to look up their pre-computed output.
        if getattr(self, "_is_offline", False):
            self._fn.current_id = test_case.id

        try:
            return await self._fn(test_case.input, test_case.context)
        except Exception as e:  # noqa: BLE001 - deliberately defensive at the boundary
            return f"Error: {e}"

    def run_sync(self, test_case: TestCase) -> str:
        """Synchronous convenience wrapper around run()."""
        return asyncio.run(self.run(test_case))


# ============================================================================
# HELPERS
# ============================================================================

def _count_positional_params(fn: Callable) -> int:
    """Best-effort count of positional parameters a callable accepts."""
    try:
        params = inspect.signature(fn).parameters.values()
        return sum(
            1 for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    except (ValueError, TypeError):
        return 2  # assume (input, context) when the signature is unavailable
