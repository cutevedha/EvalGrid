"""
evals/summarization_evals.py: Metrics for evaluating text summarization quality.

Four metrics:
  summarization_faithfulness — does the summary stay true to the source (no hallucinations)?
  summarization_conciseness  — is the summary appropriately shorter than the source?
  summarization_coverage     — what fraction of key source points appear in the summary?
  summarization_quality      — combined LLM-judge score across all three dimensions.

The source text is taken from test_case.context when set, otherwise test_case.input.
"""

import re
from typing import Set

from core.metric_registry import register_metric
from evals.llm_judge import get_llm_client, _extract_score_from_response, _JUDGE_SCORE_SCALE


# ============================================================================
# JUDGE PROMPT TEMPLATES
# ============================================================================

_FAITHFULNESS_TEMPLATE = """You are evaluating whether a summary is faithful to its source text.

Source Text:
{source}

Summary:
{summary}

A faithful summary contains only information present in the source — it does not add, invent, or contradict facts.

Score 1-5:
1 = Summary contradicts the source or invents significant information
2 = Summary contains several facts not in the source
3 = Summary is mostly faithful with a few unsupported additions
4 = Summary is faithful with only trivial deviations
5 = Summary is perfectly faithful — every statement traces back to the source

Format: REASONING: <your reasoning> SCORE: <1-5>"""

_QUALITY_TEMPLATE = """You are evaluating the overall quality of a summary.

Source Text:
{source}

Summary:
{summary}

Rate the summary holistically on a scale of 1-5 considering:
  - Faithfulness: no invented or contradicted information
  - Coverage: key points from the source are preserved
  - Conciseness: meaningfully shorter than the source without losing important content
  - Clarity: easy to read and understand

Score 1-5:
1 = Poor across all dimensions
2 = Below average — major issues with faithfulness or coverage
3 = Adequate — acceptable but with noticeable gaps
4 = Good — covers main points faithfully and concisely
5 = Excellent — faithful, concise, complete, and clear

Format: REASONING: <your reasoning> SCORE: <1-5>"""


# ============================================================================
# REGISTERED METRICS
# ============================================================================

@register_metric(
    "summarization_faithfulness",
    description="How faithfully the summary represents the source text (no hallucinations)",
    tags=["summarization", "judge", "hallucination"],
    capabilities=["generation"],
)
def summarization_faithfulness(test_case, actual_output):
    """Score whether the summary stays true to the source (1.0 = fully faithful)."""
    source = _get_source(test_case)
    if not source or not actual_output.strip():
        return {"summarization_faithfulness": 0.5}
    client = get_llm_client()
    if client is None:
        return {"summarization_faithfulness": _faithfulness_heuristic(source, actual_output)}
    prompt = _FAITHFULNESS_TEMPLATE.format(source=source, summary=actual_output)
    try:
        response = client.generate(prompt)
        score = _extract_score_from_response(response) / _JUDGE_SCORE_SCALE
        return {"summarization_faithfulness": round(score, 4)}
    except Exception:
        return {"summarization_faithfulness": _faithfulness_heuristic(source, actual_output)}


@register_metric(
    "summarization_conciseness",
    description="How concise the summary is relative to the source (ideal ratio: 10-50%)",
    tags=["summarization"],
    capabilities=["generation"],
)
def summarization_conciseness(test_case, actual_output):
    """Score 1.0 when summary is 10–50% of source length; penalise outside that band."""
    source = _get_source(test_case)
    if not source or not actual_output.strip():
        return {"summarization_conciseness": 0.0}
    return {"summarization_conciseness": _conciseness_heuristic(source, actual_output)}


@register_metric(
    "summarization_coverage",
    description="Fraction of key source points (content words) covered in the summary",
    tags=["summarization"],
    capabilities=["generation"],
)
def summarization_coverage(test_case, actual_output):
    """Score what fraction of the source's content words appear in the summary."""
    source = _get_source(test_case)
    if not source or not actual_output.strip():
        return {"summarization_coverage": 0.0}
    return {"summarization_coverage": _coverage_heuristic(source, actual_output)}


@register_metric(
    "summarization_quality",
    description="Overall summarization quality: faithfulness + coverage + conciseness via LLM judge",
    tags=["summarization", "judge"],
    capabilities=["generation"],
)
def summarization_quality(test_case, actual_output):
    """Combined quality score — LLM judge when available, heuristic average otherwise."""
    source = _get_source(test_case)
    if not source or not actual_output.strip():
        return {"summarization_quality": 0.0}
    client = get_llm_client()
    if client is None:
        faithfulness = _faithfulness_heuristic(source, actual_output)
        conciseness  = _conciseness_heuristic(source, actual_output)
        coverage     = _coverage_heuristic(source, actual_output)
        score = round((faithfulness + conciseness + coverage) / 3, 4)
        return {"summarization_quality": score}
    prompt = _QUALITY_TEMPLATE.format(source=source, summary=actual_output)
    try:
        response = client.generate(prompt)
        score = _extract_score_from_response(response) / _JUDGE_SCORE_SCALE
        return {"summarization_quality": round(score, 4)}
    except Exception:
        faithfulness = _faithfulness_heuristic(source, actual_output)
        return {"summarization_quality": faithfulness}


# ============================================================================
# HELPERS
# ============================================================================

def _get_source(test_case) -> str:
    """Return source text: context if provided, else input."""
    return (test_case.context or test_case.input or "").strip()


def _faithfulness_heuristic(source: str, summary: str) -> float:
    """Fraction of summary tokens that also appear in the source."""
    summary_tokens = set(summary.lower().split())
    source_tokens  = set(source.lower().split())
    if not summary_tokens:
        return 0.0
    return round(len(summary_tokens & source_tokens) / len(summary_tokens), 4)


def _conciseness_heuristic(source: str, summary: str) -> float:
    """
    Score 1.0 when summary length is 10–50% of source length.
    Shorter than 10% → proportional penalty; longer than 50% → proportional penalty.
    """
    source_len  = max(len(source.split()), 1)
    summary_len = len(summary.split())
    ratio = summary_len / source_len
    if ratio <= 0:
        return 0.0
    if 0.1 <= ratio <= 0.5:
        return 1.0
    if ratio < 0.1:
        return round(ratio / 0.1, 4)
    return round(max(0.0, 1.0 - (ratio - 0.5) * 2), 4)


_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "or", "but", "not", "this", "that", "it", "its", "which", "who", "also",
}


def _coverage_heuristic(source: str, summary: str) -> float:
    """Fraction of source content-words (length ≥ 4, not stopwords) covered by the summary."""
    source_words  = {w for w in re.findall(r'\b\w{4,}\b', source.lower())  if w not in _STOPWORDS}
    summary_words = {w for w in re.findall(r'\b\w{4,}\b', summary.lower()) if w not in _STOPWORDS}
    if not source_words:
        return 1.0
    return round(min(1.0, len(source_words & summary_words) / len(source_words)), 4)
