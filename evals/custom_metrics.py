# Custom Metrics - Builder classes and example metrics for user-defined evaluation
# Provides three patterns: function-based, class-based, and composite metrics

from core.metric_registry import BaseMetric, MetricRegistry, register_metric
from core.schemas import TestCase
from typing import Callable, Optional, List, Any


# ============================================================================
# CUSTOM METRIC WRAPPER (FUNCTION → CLASS BRIDGE)
# ============================================================================

class CustomMetric(BaseMetric):
    """
    Wraps a plain Python function as a registered BaseMetric.

    Use this when you already have a scoring function and want it to behave
    like a first-class metric (discoverable, taggable, versioned).
    """

    def __init__(self, name: str, compute_fn: Callable, description: str = "", tags: List[str] = None, capabilities: List[str] = None):
        """
        Args:
            name: Unique metric identifier
            compute_fn: Callable(test_case, actual_output, **kwargs) -> float
            description: Human-readable description
            tags: Tags for discovery (e.g. ["custom", "generation"])
            capabilities: AI capabilities this metric applies to
        """
        super().__init__(name, description, tags, capabilities)
        self.compute_fn = compute_fn

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        """Delegate to the wrapped function"""
        return self.compute_fn(test_case, actual_output, **kwargs)


def create_custom_metric(name: str, compute_fn: Callable, description: str = "", tags: List[str] = None, capabilities: List[str] = None) -> CustomMetric:
    """
    Helper that creates a CustomMetric and immediately registers it.

    Usage::
        def my_fn(test_case, output): return 0.8
        create_custom_metric("my_metric", my_fn, description="My metric", tags=["custom"])
    """
    metric = CustomMetric(name, compute_fn, description, tags, capabilities)
    MetricRegistry.register(metric)
    return metric


# ============================================================================
# THRESHOLD METRIC (PASS/FAIL WRAPPER)
# ============================================================================

class ThresholdMetric(BaseMetric):
    """
    Converts a continuous metric into a binary pass/fail metric.

    Computes the base metric score and returns 1.0 if it meets or exceeds
    the threshold, 0.0 otherwise.

    Example::
        ThresholdMetric("passes_bleu", base_metric_name="bleu", threshold=0.7)
    """

    def __init__(self, name: str, base_metric_name: str, threshold: float, description: str = ""):
        """
        Args:
            name: Unique name for this threshold metric
            base_metric_name: Name of the base metric to threshold
            threshold: Minimum acceptable score (inclusive)
        """
        super().__init__(name, description)
        self.base_metric_name = base_metric_name
        self.threshold = threshold

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        """Compute base metric then apply threshold"""
        base_score = MetricRegistry.compute(self.base_metric_name, test_case, actual_output, **kwargs)
        if base_score is None:
            return 0.0
        return 1.0 if base_score >= self.threshold else 0.0


# ============================================================================
# AGGREGATE METRIC (WEIGHTED COMBINATION)
# ============================================================================

class AggregateMetric(BaseMetric):
    """
    Computes a weighted average of multiple existing metrics.

    Useful when you want a single composite quality score that balances
    several independent dimensions.

    Example::
        AggregateMetric(
            "quality",
            metric_names=["bleu", "rouge_l", "f1_token_overlap"],
            weights=[0.5, 0.3, 0.2]
        )
    """

    def __init__(self, name: str, metric_names: List[str], weights: Optional[List[float]] = None, description: str = ""):
        """
        Args:
            name: Unique name for the aggregate metric
            metric_names: List of metric names to combine
            weights: Per-metric weights (defaults to equal weighting)
        """
        super().__init__(name, description)
        self.metric_names = metric_names
        # Equal weights when none provided
        if weights is None:
            self.weights = [1.0 / len(metric_names)] * len(metric_names)
        else:
            self.weights = weights

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        """Compute each sub-metric and return the weighted average"""
        scores = []
        for metric_name in self.metric_names:
            score = MetricRegistry.compute(metric_name, test_case, actual_output, **kwargs)
            if score is not None:
                scores.append(score)
            else:
                scores.append(0.0)  # Missing metric counts as 0

        if not scores:
            return 0.0

        weighted_sum = sum(s * w for s, w in zip(scores, self.weights))
        return weighted_sum / sum(self.weights[:len(scores)])


# ============================================================================
# CONDITIONAL METRIC (GATE-BASED EVALUATION)
# ============================================================================

class ConditionalMetric(BaseMetric):
    """
    Evaluates a metric only when a condition function is satisfied.

    If the condition returns False the metric returns a configurable fallback
    score instead of running the underlying evaluation.

    Example::
        ConditionalMetric(
            "fluency_if_long",
            condition_fn=lambda tc, out: len(out) > 100,
            metric_name="llm_judge_fluency",
            fallback_score=1.0  # short outputs assumed fluent
        )
    """

    def __init__(self, name: str, condition_fn: Callable, metric_name: str, fallback_score: float = 0.0, description: str = ""):
        """
        Args:
            name: Unique metric name
            condition_fn: Callable(test_case, actual_output) -> bool
            metric_name: Metric to run when condition is True
            fallback_score: Score returned when condition is False
        """
        super().__init__(name, description)
        self.condition_fn = condition_fn
        self.metric_name = metric_name
        self.fallback_score = fallback_score

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        """Run the metric if condition is met, otherwise return fallback"""
        if self.condition_fn(test_case, actual_output):
            score = MetricRegistry.compute(self.metric_name, test_case, actual_output, **kwargs)
            return score if score is not None else self.fallback_score
        return self.fallback_score


# ============================================================================
# BUILT-IN EXAMPLE CUSTOM METRICS
# ============================================================================

@register_metric("custom_length_check", description="Check if output length is within expected range", tags=["custom", "deterministic"], capabilities=["generation"])
def custom_length_check(test_case: TestCase, actual_output: str, min_length: int = 1, max_length: int = 10000) -> float:
    """
    Pass/fail check on character length of the output.

    Args:
        min_length: Minimum acceptable character length
        max_length: Maximum acceptable character length
    """
    length = len(actual_output)
    if length < min_length or length > max_length:
        return 0.0
    return 1.0


@register_metric("custom_keyword_presence", description="Check if all required keywords are present", tags=["custom", "deterministic"], capabilities=["generation", "extraction"])
def custom_keyword_presence(test_case: TestCase, actual_output: str, keywords: List[str] = None) -> float:
    """
    Fraction of required keywords found in the output.

    Args:
        keywords: List of keywords that must appear
    """
    if keywords is None:
        return 1.0
    output_lower = actual_output.lower()
    found = sum(1 for kw in keywords if kw.lower() in output_lower)
    return found / len(keywords) if keywords else 0.0


@register_metric("custom_format_check", description="Check if output matches expected format", tags=["custom", "deterministic"], capabilities=["extraction"])
def custom_format_check(test_case: TestCase, actual_output: str, format_pattern: str = None) -> float:
    """
    Regex-based format validation.

    Args:
        format_pattern: A regular expression the output must match (full match)
    """
    if format_pattern is None:
        return 1.0
    import re
    return 1.0 if re.match(format_pattern, actual_output) else 0.0
