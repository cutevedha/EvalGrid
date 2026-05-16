# Custom Metrics Guide

## Overview

EvalGrid provides three ways to create custom metrics:

1. **Decorator-based metrics** - Simple function-based metrics
2. **Class-based metrics** - Stateful metrics with complex logic
3. **Composite metrics** - Combine existing metrics

## Method 1: Decorator-Based Metrics

The simplest way to create a custom metric using the `@register_metric` decorator:

```python
from core.metric_registry import register_metric
from core.schemas import TestCase

@register_metric(
    name="my_custom_metric",
    description="My custom evaluation metric",
    tags=["custom", "generation"],
    capabilities=["generation"]
)
def my_custom_metric(test_case: TestCase, actual_output: str, **kwargs) -> float:
    # Your evaluation logic here
    if len(actual_output) > 10:
        return 1.0
    return 0.0
```

### Usage

```python
from core.orchestrator import Orchestrator

orchestrator = Orchestrator()

# Compute the metric
score = orchestrator.compute_metric(
    "my_custom_metric",
    test_case,
    actual_output
)

# List all custom metrics
custom_metrics = orchestrator.list_available_metrics(tag="custom")
```

## Method 2: Class-Based Metrics

For more complex metrics with state or multiple outputs:

```python
from core.metric_registry import BaseMetric, MetricRegistry
from core.schemas import TestCase

class MyComplexMetric(BaseMetric):
    def __init__(self):
        super().__init__(
            name="my_complex_metric",
            description="A complex metric with state",
            tags=["custom", "complex"],
            capabilities=["generation"]
        )
        self.call_count = 0

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        self.call_count += 1
        
        # Complex logic here
        score = self._evaluate(actual_output)
        return score

    def _evaluate(self, output: str) -> float:
        # Helper method
        return len(output) / 100.0

# Register the metric
metric = MyComplexMetric()
MetricRegistry.register(metric)
```

### Usage

```python
orchestrator = Orchestrator()

# Compute the metric
score = orchestrator.compute_metric(
    "my_complex_metric",
    test_case,
    actual_output
)
```

## Method 3: Composite Metrics

Combine multiple metrics with custom logic:

```python
from core.metric_registry import AggregateMetric, ThresholdMetric, MetricRegistry

# Weighted average of multiple metrics
aggregate = AggregateMetric(
    name="quality_score",
    metric_names=["bleu", "rouge_l", "f1_token_overlap"],
    weights=[0.5, 0.3, 0.2],
    description="Weighted quality score"
)
MetricRegistry.register(aggregate)

# Threshold-based metric
threshold = ThresholdMetric(
    name="passes_quality_gate",
    base_metric_name="quality_score",
    threshold=0.7,
    description="Quality score above 0.7"
)
MetricRegistry.register(threshold)
```

## Method 4: Conditional Metrics

Apply a metric only when a condition is met:

```python
from core.metric_registry import ConditionalMetric, MetricRegistry

def is_long_output(test_case, output):
    return len(output) > 100

conditional = ConditionalMetric(
    name="fluency_if_long",
    condition_fn=is_long_output,
    metric_name="llm_judge_fluency",
    fallback_score=1.0,
    description="Evaluate fluency only for long outputs"
)
MetricRegistry.register(conditional)
```

## Advanced Examples

### Example 1: Domain-Specific Metric

```python
@register_metric(
    name="medical_accuracy",
    description="Evaluate medical terminology accuracy",
    tags=["custom", "domain-specific"],
    capabilities=["generation"]
)
def medical_accuracy(test_case: TestCase, actual_output: str, **kwargs) -> float:
    medical_terms = ["diagnosis", "treatment", "symptom", "disease"]
    output_lower = actual_output.lower()
    
    found_terms = sum(1 for term in medical_terms if term in output_lower)
    return found_terms / len(medical_terms)
```

### Example 2: Multi-Step Evaluation

```python
class ComprehensiveMetric(BaseMetric):
    def __init__(self):
        super().__init__(
            name="comprehensive_eval",
            description="Multi-step evaluation",
            tags=["custom"],
            capabilities=["generation"]
        )

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        # Step 1: Check length
        length_score = 1.0 if 10 < len(actual_output) < 1000 else 0.0
        
        # Step 2: Check keywords
        keywords = kwargs.get("keywords", [])
        keyword_score = self._check_keywords(actual_output, keywords)
        
        # Step 3: Check format
        format_score = self._check_format(actual_output)
        
        # Combine scores
        return (length_score + keyword_score + format_score) / 3.0

    def _check_keywords(self, output: str, keywords: list) -> float:
        if not keywords:
            return 1.0
        found = sum(1 for kw in keywords if kw.lower() in output.lower())
        return found / len(keywords)

    def _check_format(self, output: str) -> float:
        # Check if output has proper formatting
        return 1.0 if output.startswith(output[0].upper()) else 0.5

MetricRegistry.register(ComprehensiveMetric())
```

### Example 3: Metric with External Data

```python
class FactualityMetric(BaseMetric):
    def __init__(self, knowledge_base: dict):
        super().__init__(
            name="factuality",
            description="Check against knowledge base",
            tags=["custom", "factual"],
            capabilities=["generation"]
        )
        self.knowledge_base = knowledge_base

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        facts = self.knowledge_base.get(test_case.id, [])
        
        output_lower = actual_output.lower()
        verified_facts = sum(1 for fact in facts if fact.lower() in output_lower)
        
        return verified_facts / len(facts) if facts else 0.5

# Usage
kb = {
    "test1": ["Paris is the capital of France", "France is in Europe"],
    "test2": ["Python is a programming language"],
}

metric = FactualityMetric(kb)
MetricRegistry.register(metric)
```

## Testing Custom Metrics

```python
from core.schemas import TestCase

# Create a test case
test_case = TestCase(
    id="test1",
    project="demo",
    capability="generation",
    input="Summarize this text",
    expected_output="Summary here"
)

# Test your metric
orchestrator = Orchestrator()
score = orchestrator.compute_metric(
    "my_custom_metric",
    test_case,
    "This is the actual output"
)

print(f"Score: {score}")
assert 0 <= score <= 1, "Metric must return score between 0 and 1"
```

## Best Practices

1. **Return values between 0 and 1** - Normalize scores to [0, 1] range
2. **Handle edge cases** - Empty strings, None values, etc.
3. **Document parameters** - Clearly document any kwargs your metric accepts
4. **Use descriptive names** - Make metric names self-explanatory
5. **Add tags** - Use tags for easy discovery and filtering
6. **Test thoroughly** - Test with various inputs and edge cases
7. **Keep it deterministic** - Same input should always produce same output
8. **Avoid side effects** - Don't modify test cases or global state

## Performance Considerations

For metrics that will be called frequently:

```python
class EfficientMetric(BaseMetric):
    def __init__(self):
        super().__init__(name="efficient_metric")
        self._cache = {}

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        # Cache results for repeated calls
        cache_key = (test_case.id, actual_output)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        score = self._compute_score(actual_output)
        self._cache[cache_key] = score
        return score

    def _compute_score(self, output: str) -> float:
        # Actual computation
        return len(output) / 100.0
```

## Debugging Custom Metrics

```python
@register_metric(name="debug_metric")
def debug_metric(test_case: TestCase, actual_output: str, **kwargs) -> float:
    print(f"Test ID: {test_case.id}")
    print(f"Input: {test_case.input}")
    print(f"Output: {actual_output}")
    print(f"Expected: {test_case.expected_output}")
    
    score = 1.0 if actual_output else 0.0
    print(f"Score: {score}")
    
    return score
```

## Integration with Orchestrator

```python
# Use custom metrics in evaluation
orchestrator = Orchestrator()

# List all available metrics
all_metrics = orchestrator.list_available_metrics()

# Filter by tag
custom_metrics = orchestrator.list_available_metrics(tag="custom")

# Get metadata
metadata = orchestrator.get_metric_metadata("my_custom_metric")
print(f"Description: {metadata.description}")
print(f"Tags: {metadata.tags}")
print(f"Capabilities: {metadata.capabilities}")
```

---

For the full metrics catalogue, see [Metrics Catalogue](metrics_catalogue.md).
