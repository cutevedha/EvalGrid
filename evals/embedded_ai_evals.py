# Embedded AI Evaluators - Metrics for AI components embedded inside larger applications
# Covers latency budgets, fallback behaviour, resource utilisation, and graceful degradation

from core.schemas import TestCase
from core.metric_registry import register_metric, BaseMetric, MetricRegistry
from typing import Dict, Any, Optional


# ============================================================================
# CLASS-BASED EMBEDDED AI EVALUATORS
# ============================================================================

class LatencyBudgetEvaluator(BaseMetric):
    """Checks whether a single inference call completes within an SLA budget"""

    def __init__(self, budget_ms: float = 5000):
        """
        Args:
            budget_ms: Maximum allowed latency in milliseconds (default 5 s)
        """
        super().__init__("latency_budget_compliance", "Check if latency meets budget", ["embedded_ai"], ["embedded_ai"])
        self.budget_ms = budget_ms

    def compute(self, test_case: TestCase, actual_output: str, latency_ms: float = None, **kwargs) -> float:
        """
        Args:
            latency_ms: Observed latency for this call

        Returns:
            1.0 if within budget, 0.0 otherwise
        """
        if latency_ms is None:
            return 0.0
        return 1.0 if latency_ms <= self.budget_ms else 0.0


class FallbackBehaviorEvaluator(BaseMetric):
    """Evaluates the quality of the system's fallback response when the AI component fails"""

    def __init__(self):
        super().__init__("fallback_behavior", "Evaluate fallback behavior on failure", ["embedded_ai"], ["embedded_ai"])

    def compute(self, test_case: TestCase, actual_output: str, fallback_triggered: bool = False, fallback_quality: float = 0.8, **kwargs) -> float:
        """
        If no fallback was needed, score is 1.0 (full marks).
        If a fallback was triggered, score equals the quality of the fallback response.

        Args:
            fallback_triggered: Whether the AI component failed and a fallback was used
            fallback_quality: Quality score of the fallback response (0.0–1.0)

        Returns:
            1.0 for normal operation, fallback_quality when fallback was used
        """
        if fallback_triggered:
            return fallback_quality  # Partial credit for a graceful degradation
        return 1.0  # No fallback needed — full marks


class IntegrationCorrectnessEvaluator(BaseMetric):
    """Evaluates whether the AI component integrated correctly with surrounding application code"""

    def __init__(self):
        super().__init__("integration_correctness", "Evaluate AI integration correctness", ["embedded_ai"], ["embedded_ai"])

    def compute(self, test_case: TestCase, actual_output: str, integration_status: str = "success", **kwargs) -> float:
        """
        Args:
            integration_status: "success" | "partial" | "failure"

        Returns:
            1.0 for success, 0.5 for partial, 0.0 for failure
        """
        if integration_status == "success":
            return 1.0
        elif integration_status == "partial":
            return 0.5  # Some integration points worked
        else:
            return 0.0  # Integration failed


# ============================================================================
# REGISTER CLASS-BASED EVALUATORS
# ============================================================================

_latency_budget        = LatencyBudgetEvaluator()
_fallback_behavior     = FallbackBehaviorEvaluator()
_integration_correctness = IntegrationCorrectnessEvaluator()

MetricRegistry.register(_latency_budget)
MetricRegistry.register(_fallback_behavior)
MetricRegistry.register(_integration_correctness)


# ============================================================================
# FUNCTION-BASED EMBEDDED AI METRICS
# ============================================================================

@register_metric("resource_utilization", description="Evaluate resource utilization efficiency", tags=["embedded_ai"], capabilities=["embedded_ai"])
def resource_utilization(test_case, actual_output, memory_mb: float = None, cpu_percent: float = None):
    """
    Combined memory + CPU efficiency score.

    Assumes 1 000 MB and 100% CPU are the worst acceptable values.
    Score = average(1 - memory_mb/1000, 1 - cpu_percent/100)

    Args:
        memory_mb: Peak memory usage in megabytes
        cpu_percent: Peak CPU usage as a percentage (0–100)
    """
    if memory_mb is None or cpu_percent is None:
        return {"resource_utilization": 0.5}

    memory_score = max(0.0, 1.0 - (memory_mb / 1000.0))   # 0 MB → 1.0, 1 000 MB → 0.0
    cpu_score    = max(0.0, 1.0 - (cpu_percent / 100.0))   # 0% → 1.0, 100% → 0.0
    avg_score    = (memory_score + cpu_score) / 2.0
    return {"resource_utilization": avg_score}


@register_metric("model_size_efficiency", description="Evaluate model size efficiency", tags=["embedded_ai"], capabilities=["embedded_ai"])
def model_size_efficiency(test_case, actual_output, model_size_mb: float = None, quality_score: float = None):
    """
    Quality-per-megabyte trade-off score.

    Weights quality at 70% and compactness at 30%.
    Assumes 500 MB is the upper acceptable model size.

    Args:
        model_size_mb: Size of the deployed model in megabytes
        quality_score: Quality score of the model's outputs (0.0–1.0)
    """
    if model_size_mb is None or quality_score is None:
        return {"model_size_efficiency": 0.5}

    size_score = max(0.0, 1.0 - (model_size_mb / 500.0))          # Compact model bonus
    efficiency = (quality_score * 0.7) + (size_score * 0.3)        # Weighted combination
    return {"model_size_efficiency": efficiency}


@register_metric("inference_speed", description="Evaluate inference speed", tags=["embedded_ai"], capabilities=["embedded_ai"])
def inference_speed(test_case, actual_output, inference_time_ms: float = None, target_time_ms: float = 100):
    """
    Ratio of target latency to actual latency, capped at 1.0.

    A score of 1.0 means the model ran at or faster than the target speed.

    Args:
        inference_time_ms: Actual inference time in milliseconds
        target_time_ms: Desired inference time (default 100 ms)
    """
    if inference_time_ms is None:
        return {"inference_speed": 0.0}

    speed_ratio = target_time_ms / inference_time_ms if inference_time_ms > 0 else 0.0
    return {"inference_speed": min(1.0, speed_ratio)}


@register_metric("error_recovery", description="Evaluate error recovery capability", tags=["embedded_ai"], capabilities=["embedded_ai"])
def error_recovery(test_case, actual_output, errors_encountered: int = 0, errors_recovered: int = 0):
    """
    Fraction of encountered errors from which the system successfully recovered.

    Args:
        errors_encountered: Total number of errors that occurred
        errors_recovered: Number of errors the system handled without crashing
    """
    if errors_encountered == 0:
        return {"error_recovery": 1.0}  # No errors — perfect score

    recovery_rate = errors_recovered / errors_encountered if errors_encountered > 0 else 0.0
    return {"error_recovery": recovery_rate}


@register_metric("graceful_degradation", description="Evaluate graceful degradation under load", tags=["embedded_ai"], capabilities=["embedded_ai"])
def graceful_degradation(test_case, actual_output, normal_quality: float = None, degraded_quality: float = None):
    """
    How much quality is retained when the system operates under stress or reduced resources.

    Score = 1 - (quality_drop / normal_quality).
    A score of 1.0 means zero degradation; 0.0 means total loss of quality.

    Args:
        normal_quality: Quality score under normal conditions
        degraded_quality: Quality score under degraded/stressed conditions
    """
    if normal_quality is None or degraded_quality is None:
        return {"graceful_degradation": 0.5}

    degradation = (normal_quality - degraded_quality) / normal_quality if normal_quality > 0 else 0.0
    return {"graceful_degradation": max(0.0, 1.0 - degradation)}


@register_metric("compatibility_check", description="Check compatibility with target environment", tags=["embedded_ai"], capabilities=["embedded_ai"])
def compatibility_check(test_case, actual_output, compatible: bool = True):
    """
    Binary check: does the AI component work in the target deployment environment?

    Args:
        compatible: True if compatible, False otherwise
    """
    return {"compatibility_check": 1.0 if compatible else 0.0}
