# EvalGrid - Metrics Catalogue

## Overview

EvalGrid includes 100+ built-in metrics across multiple categories for evaluating AI systems, agents, RAG pipelines, and embedded AI applications.

## Metric Categories

### 1. Deterministic Metrics

Metrics based on exact string matching and pattern recognition.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `exact_match` | Exact string match (case-sensitive) | generation, extraction | [0, 1] |
| `substring_match` | Check if expected is substring of actual | generation, extraction | [0, 1] |
| `case_insensitive_match` | Case-insensitive exact match | generation, extraction | [0, 1] |
| `numeric_tolerance` | Numeric match with tolerance | extraction | [0, 1] |
| `regex_match` | Match against regex pattern | extraction | [0, 1] |
| `contains_all_keywords` | All keywords present | generation | [0, 1] |
| `contains_any_keyword` | At least one keyword present | generation | [0, 1] |
| `excludes_keywords` | No forbidden keywords | generation | [0, 1] |
| `length_in_range` | Output length within range | generation | [0, 1] |
| `word_count_in_range` | Word count within range | generation | [0, 1] |
| `starts_with` | Output starts with prefix | generation | [0, 1] |
| `ends_with` | Output ends with suffix | generation | [0, 1] |

### 2. Semantic Metrics

Metrics based on semantic similarity and text overlap.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `semantic_similarity` | Jaccard similarity (word overlap) | generation, extraction | [0, 1] |
| `embedding_similarity` | Cosine similarity of embeddings | generation, extraction | [0, 1] |
| `bleu` | BLEU score (n-gram overlap) | generation | [0, 1] |
| `rouge_l` | ROUGE-L F1 (longest common subsequence) | generation | [0, 1] |
| `f1_token_overlap` | F1 score of token overlap | generation | [0, 1] |
| `jaccard_similarity` | Jaccard set similarity | generation | [0, 1] |

### 3. LLM Judge Metrics

Metrics using an LLM to evaluate output quality.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `llm_judge_correctness` | LLM-based correctness evaluation | generation, extraction | [0, 1] |
| `llm_judge_groundedness` | LLM-based groundedness evaluation | generation, extraction | [0, 1] |
| `llm_judge_fluency` | LLM-based fluency evaluation | generation | [0, 1] |
| `llm_judge_relevance` | LLM-based relevance evaluation | generation | [0, 1] |
| `llm_judge_helpfulness` | LLM-based helpfulness evaluation | generation | [0, 1] |
| `llm_judge_completeness` | LLM-based completeness evaluation | generation | [0, 1] |
| `llm_judge_safety` | LLM-based safety evaluation | generation | [0, 1] |

### 4. Agent Metrics

Metrics for evaluating AI agents and multi-step systems.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `tool_call_correctness` | Tool call accuracy | agent, tool_use | [0, 1] |
| `plan_coherence` | Plan step ordering and alignment | agent | [0, 1] |
| `loop_detection` | Detect repeated action sequences | agent | [0, 1] |
| `task_completion` | Agent completed the task | agent | [0, 1] |
| `context_retention` | Multi-turn context retention | agent | [0, 1] |
| `agent_latency` | Agent step latency | agent | [0, 1] |
| `handoff_quality` | Multi-agent handoff quality | multi_agent | [0, 1] |
| `step_count` | Number of steps taken | agent | [0, 1] |
| `tool_usage_rate` | Fraction of steps using tools | agent, tool_use | [0, 1] |
| `action_diversity` | Diversity of actions taken | agent | [0, 1] |

### 5. RAG Metrics

Metrics for evaluating Retrieval-Augmented Generation systems.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `faithfulness` | Answer supported by context | rag | [0, 1] |
| `context_precision` | Fraction of retrieved chunks relevant | rag | [0, 1] |
| `context_recall` | Fraction of relevant info retrieved | rag | [0, 1] |
| `answer_relevance` | Answer addresses the question | rag | [0, 1] |
| `citation_accuracy` | Accuracy of citations | rag | [0, 1] |
| `chunk_utilization` | Fraction of chunks used in answer | rag | [0, 1] |
| `retrieval_f1` | F1 score of retrieval quality | rag | [0, 1] |
| `answer_length_ratio` | Ratio of answer to context length | rag | [0, 1] |
| `retrieval_rank` | Rank of first relevant chunk | rag | [0, 1] |
| `context_coverage` | Fraction of expected output covered | rag | [0, 1] |

### 6. Safety & Toxicity Metrics

Metrics for evaluating safety and harmful content.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `toxicity_hate` | Detect hate speech | generation | [0, 1] |
| `toxicity_threat` | Detect threats | generation | [0, 1] |
| `toxicity_sexual` | Detect sexual content | generation | [0, 1] |
| `toxicity_self_harm` | Detect self-harm content | generation | [0, 1] |
| `toxicity_violence` | Detect violent content | generation | [0, 1] |
| `overall_toxicity` | Overall toxicity score | generation | [0, 1] |
| `policy_safe` | Policy compliance check | generation | [0, 1] |
| `pii_found` | PII detection | generation | [0, 1] |
| `prompt_injection_detected` | Prompt injection detection | generation | [0, 1] |

### 7. Hallucination Metrics

Metrics for detecting hallucinated or unsupported content.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `hallucination_score` | Hallucination detection | generation, rag | [0, 1] |
| `factual_grounding` | Grounding in facts | generation | [0, 1] |
| `entity_presence` | Required entities present | extraction | [0, 1] |
| `contradiction_detection` | Contradiction detection | generation | [0, 1] |
| `specificity_check` | Output specificity | generation | [0, 1] |
| `vagueness_detection` | Vague language detection | generation | [0, 1] |

### 8. Performance Metrics

Metrics for evaluating system performance and efficiency.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `latency_p50` | Median latency (ms) | agent, generation | [0, ∞) |
| `latency_p95` | 95th percentile latency (ms) | agent, generation | [0, ∞) |
| `latency_p99` | 99th percentile latency (ms) | agent, generation | [0, ∞) |
| `latency_mean` | Mean latency (ms) | agent, generation | [0, ∞) |
| `latency_max` | Maximum latency (ms) | agent, generation | [0, ∞) |
| `latency_min` | Minimum latency (ms) | agent, generation | [0, ∞) |
| `latency_stddev` | Latency standard deviation | agent, generation | [0, ∞) |
| `throughput` | Requests per second | agent, generation | [0, ∞) |
| `token_cost` | Estimated token cost (USD) | generation | [0, ∞) |
| `memory_usage` | Memory usage (MB) | agent, generation | [0, ∞) |
| `time_to_first_token` | Time to first token (ms) | generation | [0, ∞) |
| `tokens_per_second` | Generation speed | generation | [0, ∞) |
| `cost_per_1k_tokens` | Cost per 1000 tokens | generation | [0, ∞) |
| `latency_budget_compliance` | Meets latency budget | agent, generation | [0, 1] |
| `cost_efficiency` | Cost efficiency score | generation | [0, ∞) |

### 9. JSON Schema Metrics

Metrics for validating JSON output structure.

| Metric | Description | Capability | Range |
|--------|-------------|-----------|-------|
| `valid_json` | Valid JSON output | extraction | [0, 1] |
| `missing_keys` | Count of missing required keys | extraction | [0, ∞) |

## Metric Parameters

Many metrics accept optional parameters to customize behavior:

```python
# Example: numeric_tolerance with custom tolerance
orchestrator.compute_metric(
    "numeric_tolerance",
    test_case,
    actual_output,
    tolerance=0.05  # 5% tolerance
)

# Example: length_in_range with custom bounds
orchestrator.compute_metric(
    "length_in_range",
    test_case,
    actual_output,
    min_length=10,
    max_length=500
)
```

## Metric Tags

Metrics are tagged for easy discovery:

- `deterministic` - Exact matching metrics
- `semantic` - Similarity-based metrics
- `judge` - LLM-based evaluation
- `agent` - Agent-specific metrics
- `rag` - RAG system metrics
- `safety` - Safety and compliance metrics
- `toxicity` - Toxicity detection
- `hallucination` - Hallucination detection
- `performance` - Performance metrics
- `custom` - User-defined custom metrics

## Filtering Metrics

```python
# Get all safety metrics
safety_metrics = orchestrator.list_available_metrics(tag="safety")

# Get all agent metrics
agent_metrics = orchestrator.list_available_metrics(capability="agent")

# Get metrics for RAG capability
rag_metrics = orchestrator.list_available_metrics(capability="rag")
```

## Metric Metadata

Access metric information:

```python
metadata = orchestrator.get_metric_metadata("bleu")
print(metadata.description)
print(metadata.tags)
print(metadata.capabilities)
```

---

For custom metrics, see [Custom Metrics Guide](custom_metrics_guide.md).
