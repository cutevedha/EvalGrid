"""
core/metric_registry.py: The catalogue of all evaluation metrics in EvalGrid.

Think of this as the "plugin system" for metrics.  Any module can register a new
evaluation metric by using the @register_metric decorator, and any other part of
the framework can discover and run it by name: without either side needing to
know about the other.

Key concepts
------------
MetricMetadata
    Descriptive information about a metric (name, tags, which AI capabilities it
    applies to).  Used for documentation and filtering.

BaseMetric
    The class-based interface for writing a custom metric.  Subclass it, implement
    compute(), and register an instance with MetricRegistry.register().

MetricRegistry
    A singleton (one global instance) that stores every metric and exposes helpers
    to list, filter, and run them.  The Orchestrator and EvalAgent both use it.

@register_metric
    The easiest way to add a metric: decorate any function that takes
    (test_case, actual_output) and returns a dict of scores.
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from core.schemas import TestCase, EvalResult


# ============================================================================
# METRIC METADATA
# ============================================================================

@dataclass
class MetricMetadata:
    """Stores metadata about a metric for discovery and documentation"""
    name: str  # Unique metric identifier
    description: str = ""  # Human-readable description
    tags: List[str] = field(default_factory=list)  # Tags for filtering (e.g., "safety", "custom")
    capabilities: List[str] = field(default_factory=list)  # AI capabilities it evaluates
    version: str = "1.0"  # Metric version
    author: str = "unknown"  # Who created the metric
    lower_is_better: bool = False  # Whether lower scores are better (e.g., latency)


# ============================================================================
# BASE METRIC CLASS
# ============================================================================

class BaseMetric:
    """Base class for all custom metrics"""

    def __init__(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.tags = tags or []
        self.capabilities = capabilities or []
        self.metadata = MetricMetadata(
            name=name,
            description=description,
            tags=self.tags,
            capabilities=self.capabilities
        )

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        """
        Compute the metric score

        Args:
            test_case: The test case being evaluated
            actual_output: The actual output from the AI system
            **kwargs: Additional parameters specific to the metric

        Returns:
            Score between 0.0 and 1.0 (or unbounded for metrics like latency)
        """
        raise NotImplementedError("Subclasses must implement compute()")

    def __call__(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        """Allow metric to be called as a function"""
        return self.compute(test_case, actual_output, **kwargs)


# ============================================================================
# METRIC REGISTRY - SINGLETON PATTERN
# ============================================================================

class MetricRegistry:
    """
    Central registry for all metrics in the framework
    Uses singleton pattern to ensure single global instance
    """

    _instance = None  # Singleton instance
    _metrics: Dict[str, BaseMetric] = {}  # Registered metric classes
    _metric_functions: Dict[str, Callable] = {}  # Registered metric functions

    def __new__(cls):
        """Ensure only one registry instance exists"""
        if cls._instance is None:
            cls._instance = super(MetricRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, metric: BaseMetric) -> None:
        """Register a metric class"""
        registry = cls()
        registry._metrics[metric.name] = metric

    @classmethod
    def register_function(
        cls,
        name: str,
        func: Callable,
        description: str = "",
        tags: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """Register a metric function"""
        registry = cls()
        registry._metric_functions[name] = {
            "func": func,
            "metadata": MetricMetadata(
                name=name,
                description=description,
                tags=tags or [],
                capabilities=capabilities or []
            )
        }

    @classmethod
    def get(cls, name: str) -> Optional[BaseMetric]:
        """Retrieve a registered metric class by name"""
        registry = cls()
        return registry._metrics.get(name)

    @classmethod
    def get_function(cls, name: str) -> Optional[Callable]:
        """Retrieve a registered metric function by name"""
        registry = cls()
        entry = registry._metric_functions.get(name)
        return entry["func"] if entry else None

    @classmethod
    def list_metrics(cls, tag: Optional[str] = None, capability: Optional[str] = None) -> List[str]:
        """
        List all registered metrics with optional filtering

        Args:
            tag: Filter by tag (e.g., "safety", "custom")
            capability: Filter by capability (e.g., "agent", "rag")

        Returns:
            List of metric names matching the filters
        """
        registry = cls()
        metrics = list(registry._metrics.keys()) + list(registry._metric_functions.keys())

        # Filter by tag if specified
        if tag:
            filtered = []
            for m in metrics:
                if m in registry._metrics:
                    if tag in registry._metrics[m].tags:
                        filtered.append(m)
                elif m in registry._metric_functions:
                    if tag in registry._metric_functions[m]["metadata"].tags:
                        filtered.append(m)
            metrics = filtered

        # Filter by capability if specified
        if capability:
            filtered = []
            for m in metrics:
                if m in registry._metrics:
                    if capability in registry._metrics[m].capabilities:
                        filtered.append(m)
                elif m in registry._metric_functions:
                    if capability in registry._metric_functions[m]["metadata"].capabilities:
                        filtered.append(m)
            metrics = filtered

        return metrics

    @classmethod
    def get_callable(cls, name: str) -> Optional[Callable]:
        """
        Return the underlying callable for a metric (class .compute or function).

        Used by callers that need to introspect a metric's signature: e.g. to discover
        which extra data parameters it requires before deciding whether to run it.
        """
        registry = cls()
        if name in registry._metrics:
            return registry._metrics[name].compute
        if name in registry._metric_functions:
            return registry._metric_functions[name]["func"]
        return None

    @classmethod
    def get_metadata(cls, name: str) -> Optional[MetricMetadata]:
        """Get metadata for a metric"""
        registry = cls()
        if name in registry._metrics:
            return registry._metrics[name].metadata
        elif name in registry._metric_functions:
            return registry._metric_functions[name]["metadata"]
        return None

    @classmethod
    def compute(cls, name: str, test_case: TestCase, actual_output: str, **kwargs) -> Optional[float]:
        """
        Compute a metric by name

        Args:
            name: Name of the metric to compute
            test_case: The test case being evaluated
            actual_output: The actual output from the AI system
            **kwargs: Additional parameters for the metric

        Returns:
            Metric score or None if metric not found
        """
        registry = cls()
        if name in registry._metrics:
            return registry._metrics[name].compute(test_case, actual_output, **kwargs)
        elif name in registry._metric_functions:
            result = registry._metric_functions[name]["func"](test_case, actual_output, **kwargs)
            if isinstance(result, dict) and name in result:
                return result[name]
            return result
        return None


# ============================================================================
# DECORATOR FOR REGISTERING METRICS
# ============================================================================

def register_metric(
    name: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
):
    """
    Decorator to register a metric function

    Usage:
        @register_metric("my_metric", description="My custom metric", tags=["custom"])
        def my_metric(test_case, actual_output, **kwargs):
            return 0.8

    Args:
        name: Unique metric name
        description: Human-readable description
        tags: Tags for filtering
        capabilities: AI capabilities this metric evaluates
    """
    def decorator(func: Callable) -> Callable:
        MetricRegistry.register_function(name, func, description, tags, capabilities)
        return func
    return decorator
