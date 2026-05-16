# Performance Evaluators - Latency, throughput, cost, and memory metrics
# Measures how efficiently an AI system uses time, compute, and money

from core.metric_registry import register_metric, BaseMetric, MetricRegistry
from core.schemas import TestCase
from typing import List, Dict, Optional
import time
import statistics

# ============================================================================
# LATENCY EVALUATORS
# ============================================================================

class LatencyEvaluator(BaseMetric):
    """Reports the median (p50) latency across a set of measurements"""

    def __init__(self):
        super().__init__("latency_p50", "Median latency in milliseconds", ["performance"], ["agent", "generation"])

    def compute(self, test_case: TestCase, actual_output: str, latencies_ms: List[float] = None, **kwargs) -> float:
        """
        Args:
            latencies_ms: List of per-request latency measurements in milliseconds

        Returns:
            Median latency value (not normalised — use latency_budget_compliance for pass/fail)
        """
        if not latencies_ms:
            return 0.0
        p50 = statistics.median(latencies_ms)
        return p50


class LatencyPercentileEvaluator(BaseMetric):
    """Reports a configurable percentile (e.g. p95, p99) latency"""

    def __init__(self, percentile: int = 95):
        """
        Args:
            percentile: Which percentile to compute (1–99)
        """
        super().__init__(f"latency_p{percentile}", f"{percentile}th percentile latency", ["performance"], ["agent", "generation"])
        self.percentile = percentile

    def compute(self, test_case: TestCase, actual_output: str, latencies_ms: List[float] = None, **kwargs) -> float:
        """
        Args:
            latencies_ms: List of latency measurements in milliseconds

        Returns:
            Latency at the specified percentile
        """
        if not latencies_ms or len(latencies_ms) < 2:
            return 0.0
        sorted_latencies = sorted(latencies_ms)
        idx = int(len(sorted_latencies) * self.percentile / 100)
        return float(sorted_latencies[min(idx, len(sorted_latencies) - 1)])

# ============================================================================
# THROUGHPUT AND COST EVALUATORS
# ============================================================================

class ThroughputEvaluator(BaseMetric):
    """Measures how many requests the system can handle per second"""

    def __init__(self):
        super().__init__("throughput", "Requests per second", ["performance"], ["agent", "generation"])

    def compute(self, test_case: TestCase, actual_output: str, total_requests: int = None, total_time_seconds: float = None, **kwargs) -> float:
        """
        Args:
            total_requests: Number of requests completed
            total_time_seconds: Wall-clock time elapsed

        Returns:
            Requests per second (raw value, not normalised)
        """
        if not total_requests or not total_time_seconds or total_time_seconds == 0:
            return 0.0
        return total_requests / total_time_seconds


class TokenCostEvaluator(BaseMetric):
    """Estimates the USD cost of a generation based on token counts and model pricing"""

    def __init__(self):
        super().__init__("token_cost", "Estimated token cost in USD", ["performance"], ["generation"])

    def compute(self, test_case: TestCase, actual_output: str, input_tokens: int = None, output_tokens: int = None, model: str = "gpt-3.5-turbo", **kwargs) -> float:
        """
        Args:
            input_tokens: Number of prompt tokens consumed
            output_tokens: Number of completion tokens generated
            model: Model name used for pricing lookup

        Returns:
            Estimated cost in USD
        """
        if input_tokens is None or output_tokens is None:
            return 0.0

        # Per-1000-token pricing table (input, output)
        pricing = {
            "gpt-3.5-turbo":  {"input": 0.0005, "output": 0.0015},
            "gpt-4":           {"input": 0.03,   "output": 0.06},
            "claude-3-sonnet": {"input": 0.003,  "output": 0.015},
            "claude-3-opus":   {"input": 0.015,  "output": 0.075},
        }

        model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])
        cost = (input_tokens * model_pricing["input"] + output_tokens * model_pricing["output"]) / 1000
        return cost


class MemoryUsageEvaluator(BaseMetric):
    """Records peak memory usage in megabytes during inference"""

    def __init__(self):
        super().__init__("memory_usage", "Memory usage in MB", ["performance"], ["agent", "generation"])

    def compute(self, test_case: TestCase, actual_output: str, memory_mb: float = None, **kwargs) -> float:
        """
        Args:
            memory_mb: Peak memory usage measured externally (e.g. via psutil)

        Returns:
            Memory in MB (raw value)
        """
        if memory_mb is None:
            return 0.0
        return memory_mb


# ============================================================================
# REGISTER CLASS-BASED PERFORMANCE EVALUATORS
# ============================================================================

_latency_p50 = LatencyEvaluator()
_latency_p95 = LatencyPercentileEvaluator(95)
_latency_p99 = LatencyPercentileEvaluator(99)
_throughput = ThroughputEvaluator()
_token_cost = TokenCostEvaluator()
_memory_usage = MemoryUsageEvaluator()

MetricRegistry.register(_latency_p50)
MetricRegistry.register(_latency_p95)
MetricRegistry.register(_latency_p99)
MetricRegistry.register(_throughput)
MetricRegistry.register(_token_cost)
MetricRegistry.register(_memory_usage)


# ============================================================================
# FUNCTION-BASED PERFORMANCE METRICS
# ============================================================================

@register_metric("latency_mean", description="Mean latency in milliseconds", tags=["performance"], capabilities=["agent", "generation"])
def latency_mean(test_case, actual_output, latencies_ms: List[float] = None):
    """Arithmetic mean of all latency measurements"""
    if not latencies_ms:
        return {"latency_mean": 0.0}
    mean = sum(latencies_ms) / len(latencies_ms)
    return {"latency_mean": mean}


@register_metric("latency_max", description="Maximum latency in milliseconds", tags=["performance"], capabilities=["agent", "generation"])
def latency_max(test_case, actual_output, latencies_ms: List[float] = None):
    """Worst-case (maximum) latency observed"""
    if not latencies_ms:
        return {"latency_max": 0.0}
    return {"latency_max": max(latencies_ms)}


@register_metric("latency_min", description="Minimum latency in milliseconds", tags=["performance"], capabilities=["agent", "generation"])
def latency_min(test_case, actual_output, latencies_ms: List[float] = None):
    """Best-case (minimum) latency observed"""
    if not latencies_ms:
        return {"latency_min": 0.0}
    return {"latency_min": min(latencies_ms)}


@register_metric("latency_stddev", description="Standard deviation of latency", tags=["performance"], capabilities=["agent", "generation"])
def latency_stddev(test_case, actual_output, latencies_ms: List[float] = None):
    """Standard deviation of latency — high values indicate unstable performance"""
    if not latencies_ms or len(latencies_ms) < 2:
        return {"latency_stddev": 0.0}
    return {"latency_stddev": statistics.stdev(latencies_ms)}


@register_metric("time_to_first_token", description="Time to first token in streaming", tags=["performance"], capabilities=["generation"])
def time_to_first_token(test_case, actual_output, ttft_ms: float = None):
    """Time-to-first-token (TTFT) for streaming responses — critical for perceived responsiveness"""
    if ttft_ms is None:
        return {"time_to_first_token": 0.0}
    return {"time_to_first_token": ttft_ms}


@register_metric("tokens_per_second", description="Generation speed in tokens/sec", tags=["performance"], capabilities=["generation"])
def tokens_per_second(test_case, actual_output, output_tokens: int = None, generation_time_ms: float = None):
    """Decode throughput — how fast the model produces tokens"""
    if not output_tokens or not generation_time_ms or generation_time_ms == 0:
        return {"tokens_per_second": 0.0}
    tps = output_tokens / (generation_time_ms / 1000)
    return {"tokens_per_second": tps}


@register_metric("cost_per_1k_tokens", description="Cost per 1000 tokens", tags=["performance"], capabilities=["generation"])
def cost_per_1k_tokens(test_case, actual_output, total_cost: float = None, total_tokens: int = None):
    """Normalised cost — useful for comparing models across different call sizes"""
    if not total_cost or not total_tokens or total_tokens == 0:
        return {"cost_per_1k_tokens": 0.0}
    cost_per_1k = (total_cost / total_tokens) * 1000
    return {"cost_per_1k_tokens": cost_per_1k}


@register_metric("latency_budget_compliance", description="Check if latency meets budget", tags=["performance"], capabilities=["agent", "generation"])
def latency_budget_compliance(test_case, actual_output, latency_ms: float = None, budget_ms: float = 5000):
    """
    Binary pass/fail check against a latency SLA

    Args:
        latency_ms: Observed latency for this request
        budget_ms: Maximum allowed latency (default 5 000 ms)

    Returns:
        1.0 if within budget, 0.0 otherwise
    """
    if latency_ms is None:
        return {"latency_budget_compliance": 0.0}
    return {"latency_budget_compliance": 1.0 if latency_ms <= budget_ms else 0.0}


@register_metric("cost_efficiency", description="Cost efficiency score (lower is better)", tags=["performance"], capabilities=["generation"])
def cost_efficiency(test_case, actual_output, total_cost: float = None, output_quality_score: float = None):
    """
    Quality-per-dollar ratio

    A higher score means more output quality was obtained per unit of cost.
    Capped at 1.0 for normalisation.
    """
    if total_cost is None or output_quality_score is None or output_quality_score == 0:
        return {"cost_efficiency": 0.0}
    efficiency = output_quality_score / total_cost if total_cost > 0 else 0.0
    return {"cost_efficiency": min(1.0, efficiency)}
