# EvalGrid

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, production-grade framework for evaluating and testing AI systems, agents, RAG pipelines, and embedded AI applications with 100+ built-in metrics and custom metric support.

## Features

### Core Capabilities
- **100+ Built-in Metrics** across 9 categories (deterministic, semantic, judge, agent, RAG, safety, performance, robustness, fairness)
- **Custom Metric Builder** - Decorator-based and class-based custom metrics with full registry support
- **Multi-Agent Evaluation** - Handoff quality, orchestration correctness, communication clarity
- **RAG Evaluation** - Faithfulness, context precision/recall, citation accuracy, answer relevance
- **Agent Evaluation** - Tool call correctness, plan coherence, loop detection, task completion
- **Embedded AI Evaluation** - Latency budgets, fallback behavior, resource utilization, graceful degradation
- **Safety & Compliance** - Toxicity detection, hallucination detection, PII masking, prompt injection detection
- **Robustness & Fairness** - Adversarial robustness, demographic parity, equalized odds, bias detection
- **Performance Metrics** - Latency percentiles, throughput, token cost, memory usage
- **Async-First Architecture** - Batch evaluation with configurable concurrency
- **Rich Reporting** - HTML dashboards, CSV scorecards, JSON exports, Markdown reports
- **CI/CD Integration** - Gate-based evaluation with configurable thresholds per severity

### Supported AI Types
- Standalone LLM applications
- Multi-step AI agents
- Multi-agent systems
- Retrieval-Augmented Generation (RAG)
- Embedded AI in larger applications
- Tool-using agents
- Classification systems
- Generation systems
- Extraction systems

## Quick Start

### Installation

```bash
pip install -e .
```

### Run Demo

```bash
eval-grid run-demo
```

This generates:
- `output/scorecard.csv` - Test results in CSV format
- `output/report.html` - Rich HTML dashboard
- `output/run_results.json` - Structured JSON results
- `output/report.md` - Markdown report for PR comments

### List Available Metrics

```bash
eval-grid list-metrics
eval-grid list-metrics --tag safety
eval-grid list-metrics --capability agent
```

## Usage Examples

### Basic Evaluation

```python
from core.schemas import TestCase
from core.orchestrator import Orchestrator

# Create orchestrator
orchestrator = Orchestrator()

# Create test case
test_case = TestCase(
    id="test1",
    project="my_project",
    capability="generation",
    input="Summarize AI",
    expected_output="AI is artificial intelligence",
    severity="high"
)

# Run evaluation
result = orchestrator.run(test_case, "AI is artificial intelligence")

print(f"Passed: {result.passed}")
print(f"Scores: {result.scores}")
```

### Agent Evaluation

```python
from core.schemas import AgentTestCase, ToolCall, AgentStep, AgentTrace

# Create agent test case
test_case = AgentTestCase(
    id="agent1",
    project="my_project",
    capability="agent",
    input="Search for AI news",
    tools_available=["search", "summarize"],
    expected_tool_calls=[
        ToolCall(name="search", parameters={"query": "AI news"})
    ],
    expected_plan=["search", "summarize"],
    max_steps=5
)

# Create agent trace
trace = AgentTrace(
    agent_id="agent1",
    steps=[
        AgentStep(step_number=1, action="search", tool_calls=[...]),
        AgentStep(step_number=2, action="summarize", tool_calls=[...])
    ],
    success=True
)

# Evaluate
result = orchestrator.run(test_case, "AI news summary")
```

### RAG Evaluation

```python
from core.schemas import RAGTestCase

test_case = RAGTestCase(
    id="rag1",
    project="my_project",
    capability="rag",
    input="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI",
        "ML enables systems to learn from data"
    ],
    expected_output="Machine learning is a subset of AI that enables systems to learn from data",
    expected_citations=[0, 1]
)

result = orchestrator.run(test_case, actual_output)
```

### Batch Evaluation

```python
from pipelines.batch_runner import BatchRunner

# Create batch runner
runner = BatchRunner(orchestrator, concurrency=5)

# Run batch
test_cases = [test_case1, test_case2, test_case3]
outputs = {
    "test1": "output1",
    "test2": "output2",
    "test3": "output3"
}

results = runner.run_batch(test_cases, outputs)

# Get metrics
print(f"Pass rate: {runner.get_pass_rate():.1%}")
print(f"Metrics: {runner.get_metrics()}")
```

### Custom Metrics

#### Decorator-Based

```python
from core.metric_registry import register_metric

@register_metric(
    name="my_metric",
    description="My custom metric",
    tags=["custom"],
    capabilities=["generation"]
)
def my_metric(test_case, actual_output, **kwargs) -> float:
    if len(actual_output) > 10:
        return 1.0
    return 0.0

# Use it
score = orchestrator.compute_metric("my_metric", test_case, output)
```

#### Class-Based

```python
from core.metric_registry import BaseMetric, MetricRegistry

class MyComplexMetric(BaseMetric):
    def __init__(self):
        super().__init__(
            name="complex_metric",
            description="Complex metric",
            tags=["custom"],
            capabilities=["generation"]
        )

    def compute(self, test_case, actual_output, **kwargs) -> float:
        # Your logic here
        return 0.8

MetricRegistry.register(MyComplexMetric())
```

### Dataset Management

```python
from synthetic.dataset_builder import DatasetBuilder

# Create dataset
builder = DatasetBuilder("my_dataset", "My test dataset")

# Add test cases
builder.add_test_case({
    "id": "test1",
    "input": "test input",
    "expected_output": "test output",
    "capability": "generation",
    "severity": "high"
})

# Save
builder.save_json("datasets/my_dataset.json")

# Load
builder.load_json("datasets/my_dataset.json")

# Filter and analyze
gen_cases = builder.filter_by_capability("generation")
stats = builder.get_statistics()
```

### Data Augmentation

```python
from synthetic.augmentation import augment_dataset, paraphrase_text, inject_typos

# Augment dataset
test_cases = [...]
augmented = augment_dataset(test_cases, augmentation_factor=3)

# Paraphrase
paraphrased = paraphrase_text("This is a test")

# Add typos
with_typos = inject_typos("This is a test", error_rate=0.1)
```

### Red Team Generation

```python
from synthetic.redteam import generate_redteam_cases, generate_category_attacks

# Generate all red team cases
all_attacks = generate_redteam_cases()

# Generate specific category
prompt_injection_attacks = generate_category_attacks("prompt_injection", count=5)
```

### Reporting

```python
from reports.html_report import generate_rich_html_report
from reports.json_report import generate_json_report
from reports.markdown_report import save_markdown_report

# HTML report
generate_rich_html_report(results, "report.html", title="My Report")

# JSON report
generate_json_report(results, "results.json")

# Markdown report
save_markdown_report(results, "report.md")
```

### CLI Commands

```bash
# Run demo
eval-grid run-demo

# List metrics
eval-grid list-metrics
eval-grid list-metrics --tag safety
eval-grid list-metrics --capability agent

# Export results
eval-grid export --format html --input results.json --output report.html
eval-grid export --format markdown --input results.json --output report.md

# Compare runs
eval-grid compare --baseline baseline.json --current current.json --output comparison.md
```

## Metrics Catalogue

### Deterministic Metrics (12)
- `exact_match`, `substring_match`, `case_insensitive_match`, `numeric_tolerance`
- `regex_match`, `contains_all_keywords`, `contains_any_keyword`, `excludes_keywords`
- `length_in_range`, `word_count_in_range`, `starts_with`, `ends_with`

### Semantic Metrics (6)
- `semantic_similarity`, `embedding_similarity`, `bleu`, `rouge_l`
- `f1_token_overlap`, `jaccard_similarity`

### LLM Judge Metrics (7)
- `llm_judge_correctness`, `llm_judge_groundedness`, `llm_judge_fluency`
- `llm_judge_relevance`, `llm_judge_helpfulness`, `llm_judge_completeness`, `llm_judge_safety`

### Agent Metrics (10)
- `tool_call_correctness`, `plan_coherence`, `loop_detection`, `task_completion`
- `context_retention`, `agent_latency`, `handoff_quality`, `step_count`
- `tool_usage_rate`, `action_diversity`

### RAG Metrics (10)
- `faithfulness`, `context_precision`, `context_recall`, `answer_relevance`
- `citation_accuracy`, `chunk_utilization`, `retrieval_f1`, `answer_length_ratio`
- `retrieval_rank`, `context_coverage`

### Safety Metrics (9)
- `toxicity_hate`, `toxicity_threat`, `toxicity_sexual`, `toxicity_self_harm`, `toxicity_violence`
- `overall_toxicity`, `policy_safe`, `pii_found`, `prompt_injection_detected`

### Hallucination Metrics (6)
- `hallucination_score`, `factual_grounding`, `entity_presence`
- `contradiction_detection`, `specificity_check`, `vagueness_detection`

### Performance Metrics (15)
- `latency_p50`, `latency_p95`, `latency_p99`, `latency_mean`, `latency_max`, `latency_min`, `latency_stddev`
- `throughput`, `token_cost`, `memory_usage`, `time_to_first_token`, `tokens_per_second`
- `cost_per_1k_tokens`, `latency_budget_compliance`, `cost_efficiency`

### Robustness Metrics (7)
- `robustness_score`, `consistency_under_paraphrase`, `typo_robustness`
- `adversarial_robustness`, `demographic_parity`, `equalized_odds`, `calibration`

### Fairness & Bias Metrics (15)
- `bias_detection`, `fairness_score`, `stereotype_detection`, `representation_balance`
- `gender_bias_detection`, `age_bias_detection`, `cultural_sensitivity`, `inclusivity_score`
- `disparate_impact`, `equal_opportunity`, `predictive_parity`, `counterfactual_fairness`
- `multi_agent_handoff`, `orchestrator_correctness`, `agent_communication_clarity`

See [Metrics Catalogue](docs/metrics_catalogue.md) for complete documentation.

## Custom Metrics

Create custom metrics for your specific needs:

```python
@register_metric("domain_specific", tags=["custom"])
def domain_specific(test_case, output, **kwargs):
    # Your evaluation logic
    return score  # 0.0 to 1.0
```

See [Custom Metrics Guide](docs/custom_metrics_guide.md) for advanced examples.

## Architecture

```
eval_grid/
├── core/                    # Core schemas, orchestrator, metric registry
├── evals/                   # 100+ evaluation metrics
├── guards/                  # Safety guards (PII, toxicity, hallucination)
├── adapters/                # LLM adapters (OpenAI, Anthropic, Ollama)
├── pipelines/               # Batch, streaming, gating runners
├── reports/                 # HTML, JSON, CSV, Markdown reports
├── synthetic/               # Data augmentation, red team, dataset builder
├── scripts/                 # CLI interface
├── tests/                   # Comprehensive test suite
└── docs/                    # Documentation
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_comprehensive.py::TestMetricRegistry -v

# Run with coverage
pytest tests/ --cov=core --cov=evals
```

## Configuration

### LLM Judge Setup

```python
from adapters.llm.openai_adapter import OpenAIAdapter
from evals.llm_judge import set_llm_client

# Configure LLM client
client = OpenAIAdapter(api_key="sk-...", model="gpt-4")
set_llm_client(client)

# Now LLM judge metrics will use real LLM
result = orchestrator.run(test_case, output)
```

### Embedder Setup

```python
from evals.semantic import set_embedder, SentenceTransformerEmbedder

# Use sentence transformers
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
set_embedder(embedder)

# Now embedding_similarity will use real embeddings
score = orchestrator.compute_metric("embedding_similarity", test_case, output)
```

## Performance

- **Async-first** - Batch evaluation with configurable concurrency
- **Lazy loading** - Optional dependencies loaded only when needed
- **Caching** - Results cached to avoid redundant computations
- **Streaming** - Real-time evaluation with callbacks

## Best Practices

1. **Start with deterministic metrics** - Fast, reliable baseline
2. **Add semantic metrics** - Better coverage of meaning
3. **Use LLM judge sparingly** - More expensive, use for final validation
4. **Create custom metrics** - Domain-specific evaluation
5. **Test robustness** - Evaluate against adversarial inputs
6. **Monitor fairness** - Check for bias across demographic groups
7. **Set thresholds** - Define pass/fail criteria per severity level
8. **Iterate** - Use results to improve your AI system

## Contributing

Contributions welcome! Areas for enhancement:
- Additional metric implementations
- New LLM adapters
- Performance optimizations
- Documentation improvements

## License

MIT License - see [LICENSE](LICENSE) for details

## Support

- 📖 [Full Documentation](docs/)
- 📊 [Metrics Catalogue](docs/metrics_catalogue.md)
- 🔧 [Custom Metrics Guide](docs/custom_metrics_guide.md)
- 🐛 [Report Issues](https://github.com/your-repo/issues)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{eval_grid,
  title={EvalGrid},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

---

**Version:** 1.0.0  
**Last Updated:** 2024

