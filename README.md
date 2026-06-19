<div align="center">

# EvalGrid

**The fastest, most cost-efficient LLM evaluation framework.**

100+ metrics · async parallel evaluation · batched LLM judging · pytest-native · zero-config quickstart

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-349%20passing-brightgreen)]()
[![Token Reduction](https://img.shields.io/badge/token%20use-81%25%20less-orange)]()
[![Speedup](https://img.shields.io/badge/speedup-20x-success)]()
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

---

## Why EvalGrid

| | DeepEval | RAGAS | **EvalGrid** |
|---|---|---|---|
| Built-in metrics | ~14 | ~8 | **100+** |
| One-line `evaluate()` API | ✓ | ✓ | ✓ |
| Pytest `assert_test()` | ✓ | ✗ | ✓ |
| Parallel async evaluation | ✓ | partial | ✓ (20x speedup) |
| **Batched multi-rubric judging** | ✗ | ✗ | ✓ **(80% fewer tokens)** |
| Multi-format data loader (Excel/CSV/JSON/JSONL/YAML) | ✗ | ✗ | ✓ |
| Autonomous adaptive eval agent | ✗ | ✗ | ✓ |
| Governance pipeline + audit trail | ✗ | ✗ | ✓ |
| Real LLM judge auto-detection from env | partial | ✗ | ✓ |
| Cost tracking per metric | ✗ | ✗ | ✓ |

---

## Install

```bash
pip install evalgrid

# With your preferred LLM provider:
pip install "evalgrid[openai]"        # OpenAI
pip install "evalgrid[anthropic]"     # Anthropic Claude
pip install "evalgrid[gemini]"        # Google Gemini
pip install "evalgrid[all]"           # everything
```

---

## 30-second quickstart

```python
from evalgrid import evaluate

run = evaluate(
    cases=[
        {"input": "What is the capital of France?",
         "output": "Paris is the capital of France.",
         "expected_output": "The capital of France is Paris."},
    ],
    metrics="rag",   # preset bundle
)

print(run.summary())
run.to_html("report.html")
```

That's it. EvalGrid auto-detects `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` from your environment and uses real LLM judges. No setup files, no boilerplate.

---

## Why it's faster

### 20x speedup via async parallel evaluation

```python
# 200 cases × 3 LLM-judge metrics @ 500ms/call
# Sequential:        ~5 minutes
# EvalGrid default:  15 seconds  (20x faster)
```

Set `concurrency=25` (default 10) and EvalGrid runs cases in parallel with semaphore-based rate limiting.

### 80% token reduction via batched judging

When you request multiple LLM-judge metrics, EvalGrid scores them all in **ONE LLM call** per case instead of N calls:

```python
# 100 cases × 5 LLM-judge metrics

# Without batching:  500 calls   ·  106,000 tokens  ·  $0.0229
# With batching:     100 calls   ·   19,800 tokens  ·  $0.0036

#                    80% fewer calls    81% fewer tokens   84% cheaper
```

Enabled by default. Zero code changes required.

---

## Pytest integration

```python
from evalgrid import assert_test

def test_my_chatbot():
    assert_test(
        input="What is AI?",
        output=my_chatbot("What is AI?"),
        expected="AI is artificial intelligence.",
        metrics=["correctness", "relevance"],
        threshold=0.7,
    )
```

Failed assertions show exactly which metric failed and its score.

---

## Load datasets in any format

```python
from evalgrid import evaluate

# Excel
evaluate(cases="tests.xlsx", metrics="rag")

# JSON
evaluate(cases="tests.json", metrics="safety")

# CSV with custom column names (auto-aliased)
evaluate(cases="qa_pairs.csv", metrics="generation")

# YAML
evaluate(cases="redteam.yaml", metrics="adversarial")
```

Column aliases recognised automatically:
- `question` / `prompt` / `query` → `input`
- `answer` / `reference` / `ground_truth` → `expected_output`
- `documents` / `context` / `passage` → `context`
- ...and 20+ more

---

## Metric presets

```python
from evalgrid import evaluate, MetricSet

evaluate(cases, metrics=MetricSet.RAG)             # context_precision, recall, faithfulness, ...
evaluate(cases, metrics=MetricSet.SAFETY)          # all guardrails: hate, threat, illegal, ...
evaluate(cases, metrics=MetricSet.GENERATION)      # correctness, relevance, fluency, ...
evaluate(cases, metrics=MetricSet.SUMMARIZATION)   # faithfulness, conciseness, coverage
evaluate(cases, metrics=MetricSet.STRUCTURED)      # json_correctness, exact_match, ...
evaluate(cases, metrics=MetricSet.AGENT)           # tool calls, task success, token budget
evaluate(cases, metrics=MetricSet.BIAS)            # demographic_parity, equal_opportunity
evaluate(cases, metrics=MetricSet.ROBUSTNESS)      # paraphrase, typo, adversarial
evaluate(cases, metrics=MetricSet.REFERENCE)       # gold-answer comparison
```

Strings work too:

```python
evaluate(cases, metrics="rag")
evaluate(cases, metrics="safety")
```

---

## Real LLM judges, out of the box

```python
from evalgrid import configure, evaluate

# Explicit model
configure(judge="gpt-4o-mini")
evaluate(cases, metrics="generation")

# Explicit with API key + custom endpoint
configure(
    judge="gpt-4o",
    api_key="sk-...",
    base_url="https://my.azure.openai.com",
    temperature=0,
)

# Or just set env var and EvalGrid auto-detects
# export OPENAI_API_KEY=sk-...
evaluate(cases, metrics="generation")
```

Auto-detection priority: `EVALGRID_JUDGE_MODEL` > `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `GEMINI_API_KEY`.

---

## CLI

```bash
# Scaffold a sample project
eval-grid init

# Run the bundled quickstart demo end-to-end + open HTML report
eval-grid quickstart

# Evaluate a dataset file with a preset
eval-grid eval --cases tests.xlsx --metrics rag --threshold 0.7

# List every registered metric
eval-grid list-metrics

# Autonomous adaptive evaluation
eval-grid auto --goal "test refusal of harmful prompts" --target openai

# Governed evaluation (6-step audit pipeline)
eval-grid govern --goal "production launch safety check" --data-file tests.xlsx
```

---

## Custom metrics

```python
from evalgrid import evaluate
from core.metric_registry import register_metric

@register_metric("my_custom_metric", description="Domain-specific score", tags=["custom"])
def my_metric(test_case, actual_output):
    score = compute_my_score(test_case.input, actual_output)
    return {"my_custom_metric": score}

evaluate(cases, metrics=["my_custom_metric"])
```

---

## G-Eval — define your own judge rubric in plain English

```python
from evalgrid.evals.structured_evals import GEvalMetric

GEvalMetric(
    name="insurance_response_quality",
    rubric_description="Evaluate an insurance chatbot response",
    evaluation_steps=[
        "Does the response acknowledge the customer's concern empathetically?",
        "Does it provide accurate information about the claims process?",
        "Does it avoid making unauthorised commitments?",
        "Does it direct the customer to appropriate next steps?",
    ],
).as_metric()

evaluate(cases, metrics=["insurance_response_quality"])
```

---

## Governance & audit

For regulated or production workflows:

```python
from evalgrid.governance import GovernancePipeline, EvalObjective, AcceptancePolicy

pipeline = GovernancePipeline(
    EvalObjective(suite="production", objective="safety launch gate"),
    AcceptancePolicy(min_sample_size=30)
        .add_gate("policy_safe", 1.0, tier="critical")
        .add_gate("refused", 1.0, tier="exploratory"),
)

outcome = pipeline.run(samples, runner, scorer)
# outcome.blocked, outcome.audit, outcome.report
```

Built-in: dataset versioning · judge prompt versioning · bias/leakage detection · red-flag audit log.

---

## Headline numbers

- ✅ **349 tests passing** (every public API is covered)
- 🚀 **20x parallel speedup** on real-world workloads
- 💸 **81% fewer tokens** vs single-rubric judging
- 🧠 **100+ built-in metrics** — generation, RAG, safety, agent, bias, robustness, perf
- 🔌 **5 file formats** for test data — Excel, JSON, JSONL, CSV, YAML
- 🛡️ **4 LLM providers** out of the box — OpenAI, Anthropic, Gemini, Ollama
- 📊 **Beautiful HTML reports** with per-case scores, cost tracking, judge usage

---

## Documentation

- [Quickstart](#30-second-quickstart)
- [API reference](docs/api.md) *(coming soon)*
- [Metric catalog](docs/metrics.md) *(coming soon)*
- [Migration from DeepEval](docs/migrate-deepeval.md) *(coming soon)*
- [Contribution guide](CONTRIBUTING.md) *(coming soon)*

---

## License

MIT © Saro

---

<div align="center">
<sub>Built to be the evaluation framework you actually enjoy using.</sub>
</div>
