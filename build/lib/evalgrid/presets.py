"""
evalgrid/presets.py: Pre-built metric bundles for common evaluation scenarios.

Instead of memorising 100+ metric names, just pick a preset:

    from evalgrid import evaluate, MetricSet

    evaluate(cases, metrics=MetricSet.RAG)         # ← 5 metrics for RAG quality
    evaluate(cases, metrics=MetricSet.SAFETY)      # ← all guardrails
    evaluate(cases, metrics=MetricSet.GENERATION)  # ← correctness/relevance/fluency
    evaluate(cases, metrics="rag")                 # ← string form also works

Presets are simple lists of metric names; you can combine and extend them:

    custom = MetricSet.RAG + MetricSet.SAFETY + ["my_custom_metric"]
"""

from typing import Dict, List


class MetricSet:
    """Curated bundles of metrics for common evaluation scenarios."""

    # ── Core generation quality (the "every-day" preset) ────────────────────
    GENERATION: List[str] = [
        "llm_judge_correctness",
        "llm_judge_relevance",
        "llm_judge_fluency",
        "llm_judge_helpfulness",
        "llm_judge_completeness",
    ]

    # ── Retrieval-Augmented Generation ──────────────────────────────────────
    RAG: List[str] = [
        "context_precision",
        "context_recall",
        "context_relevancy",
        "llm_judge_groundedness",
        "hallucination_score",
        "retrieval_f1",
    ]

    # ── Safety / guardrails ─────────────────────────────────────────────────
    SAFETY: List[str] = [
        "overall_toxicity",
        "toxicity_hate",
        "toxicity_threat",
        "toxicity_violence",
        "toxicity_self_harm",
        "toxicity_sexual",
        "toxicity_illegal_activity",
        "toxicity_politics",
        "toxicity_religion",
        "toxicity_medical_advice",
        "toxicity_score_continuous",
    ]

    # ── Adversarial / red-team ──────────────────────────────────────────────
    ADVERSARIAL: List[str] = [
        "refusal_quality",
        "behavior_correctness",
        "adversarial_robustness",
        "overall_toxicity",
    ]

    # ── Summarization ───────────────────────────────────────────────────────
    SUMMARIZATION: List[str] = [
        "summarization_faithfulness",
        "summarization_conciseness",
        "summarization_coverage",
        "summarization_quality",
    ]

    # ── Structured output (JSON, extraction) ────────────────────────────────
    STRUCTURED: List[str] = [
        "json_correctness",
        "exact_match",
        "substring_match",
        "case_insensitive_match",
    ]

    # ── Agentic systems ─────────────────────────────────────────────────────
    AGENT: List[str] = [
        "task_success_rate",
        "tool_call_error_rate",
        "tool_usage_rate",
        "llm_calls_per_task",
        "tokens_per_task",
        "context_window_utilization",
        "max_iteration_reached",
        "step_count",
    ]

    # ── Performance / cost / reliability ────────────────────────────────────
    PERFORMANCE: List[str] = [
        "latency_mean",
        "latency_max",
        "tokens_per_second",
        "cost_per_1k_tokens",
        "cost_efficiency",
        "cache_hit_rate",
        "provider_error_rate",
        "timeout_rate",
        "retry_rate",
    ]

    # ── Bias / fairness ─────────────────────────────────────────────────────
    BIAS: List[str] = [
        "gender_bias_detection",
        "age_bias_detection",
        "stereotype_detection",
        "demographic_parity",
        "equal_opportunity",
        "counterfactual_fairness",
        "cultural_sensitivity",
        "inclusivity_score",
    ]

    # ── Robustness ──────────────────────────────────────────────────────────
    ROBUSTNESS: List[str] = [
        "typo_robustness",
        "consistency_under_paraphrase",
        "adversarial_robustness",
        "graceful_degradation",
        "error_recovery",
    ]

    # ── Reference-based (gold answer required) ──────────────────────────────
    REFERENCE: List[str] = [
        "llm_judge_reference_correctness",
        "behavior_correctness",
        "exact_match",
        "f1_token_overlap",
        "rouge_l",
        "bleu",
        "embedding_similarity",
    ]

    # ── Everything (use sparingly — expensive) ──────────────────────────────
    ALL: str = "__all__"  # Sentinel; expanded by evaluator to every registered metric.


# Mapping from string aliases → preset lists
_PRESET_ALIASES: Dict[str, List[str]] = {
    "generation":     MetricSet.GENERATION,
    "rag":            MetricSet.RAG,
    "safety":         MetricSet.SAFETY,
    "adversarial":    MetricSet.ADVERSARIAL,
    "red_team":       MetricSet.ADVERSARIAL,
    "redteam":        MetricSet.ADVERSARIAL,
    "summarization":  MetricSet.SUMMARIZATION,
    "summary":        MetricSet.SUMMARIZATION,
    "structured":     MetricSet.STRUCTURED,
    "extraction":     MetricSet.STRUCTURED,
    "json":           MetricSet.STRUCTURED,
    "agent":          MetricSet.AGENT,
    "agentic":        MetricSet.AGENT,
    "performance":    MetricSet.PERFORMANCE,
    "perf":           MetricSet.PERFORMANCE,
    "cost":           MetricSet.PERFORMANCE,
    "bias":           MetricSet.BIAS,
    "fairness":       MetricSet.BIAS,
    "robustness":     MetricSet.ROBUSTNESS,
    "reference":      MetricSet.REFERENCE,
    "gold":           MetricSet.REFERENCE,
    "ground_truth":   MetricSet.REFERENCE,
    "all":            MetricSet.ALL,
    "everything":     MetricSet.ALL,
}


def resolve_preset(name: str) -> List[str]:
    """Look up a preset by its string alias. Raises ValueError if unknown."""
    key = name.lower().strip().replace("-", "_").replace(" ", "_")
    if key not in _PRESET_ALIASES:
        available = ", ".join(sorted(set(_PRESET_ALIASES.keys())))
        raise ValueError(
            f"Unknown metric preset '{name}'.\n"
            f"Available presets: {available}\n"
            f"You can also pass an explicit list of metric names."
        )
    return _PRESET_ALIASES[key]
