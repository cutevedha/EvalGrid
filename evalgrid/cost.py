"""
evalgrid/cost.py: Lightweight token + dollar accounting for evaluation runs.

The framework cannot read every adapter's billing records, but it can track:
  • how many LLM calls each metric made
  • approximate input/output tokens (via a configurable counter)
  • estimated cost using a pricing table

Usage
-----
    from evalgrid import CostTracker

    tracker = CostTracker(model="gpt-4o-mini")
    evaluate(cases, metrics=["correctness"], cost_tracker=tracker)
    print(tracker.summary())
    # { calls: 12, input_tokens: 3210, output_tokens: 480, cost_usd: 0.0024 }
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


# Standard published prices (per 1K tokens). Override per-instance when needed.
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o":           {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini":      {"input": 0.00015, "output": 0.0006},
    "gpt-4":            {"input": 0.03,   "output": 0.06},
    "gpt-3.5-turbo":    {"input": 0.0005, "output": 0.0015},
    "claude-3-opus":    {"input": 0.015,  "output": 0.075},
    "claude-3-sonnet":  {"input": 0.003,  "output": 0.015},
    "claude-3-haiku":   {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-6": {"input": 0.003,  "output": 0.015},
}


@dataclass
class CostTracker:
    """Accumulates LLM call statistics across an evaluation run."""

    model: str = "gpt-4o-mini"
    pricing: Dict[str, Dict[str, float]] = field(default_factory=lambda: DEFAULT_PRICING.copy())
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_metric: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        metric_name: str,
        input_text: str = "",
        output_text: str = "",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """
        Log one LLM call. Token counts default to a simple word-count proxy
        (good enough for budgeting; replace with a real tokenizer if needed).
        """
        in_tok  = input_tokens  if input_tokens  is not None else _approx_tokens(input_text)
        out_tok = output_tokens if output_tokens is not None else _approx_tokens(output_text)
        with self._lock:
            self.calls += 1
            self.input_tokens  += in_tok
            self.output_tokens += out_tok
            bucket = self.by_metric.setdefault(metric_name, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            bucket["calls"] += 1
            bucket["input_tokens"]  += in_tok
            bucket["output_tokens"] += out_tok

    def estimated_cost_usd(self) -> float:
        """Return total cost in USD using the configured pricing table."""
        rates = self.pricing.get(self.model)
        if not rates:
            return 0.0
        cost = (self.input_tokens / 1000) * rates["input"] + (self.output_tokens / 1000) * rates["output"]
        return round(cost, 6)

    def summary(self) -> Dict:
        """Return a JSON-serialisable summary of all tracked usage."""
        return {
            "model":         self.model,
            "calls":         self.calls,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.input_tokens + self.output_tokens,
            "cost_usd":      self.estimated_cost_usd(),
            "by_metric":     self.by_metric,
        }


def _approx_tokens(text: str) -> int:
    """Approximate token count: ~1.3 tokens per word on average."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)
