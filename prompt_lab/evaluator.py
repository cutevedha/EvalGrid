"""
prompt_lab/evaluator.py: Score LLM responses and suggest prompt improvements.

Scoring rubric (each dimension 0–10, equal weight):
  - Relevance    – did the response answer the actual prompt?
  - Completeness – are all key points covered?
  - Clarity      – is the response well-structured and easy to read?
  - Format       – does it follow the format requested in the prompt (bullets, table, etc.)?
  - Actionable   – could a QA engineer use this output directly?

The judge is Claude (claude-haiku-4-5) to keep cost low.
A summary verdict is produced for each LLM:
  PASS  (avg >= 7.0)  | PARTIAL (avg 5.0–6.9) | FAIL (avg < 5.0)

When any LLM scores FAIL/PARTIAL, a fixed prompt is generated automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prompt_lab.runner import LLMResult


PASS_THRESHOLD = 7.0
PARTIAL_THRESHOLD = 5.0

DIMENSIONS = ["relevance", "completeness", "clarity", "format", "actionable"]

SCORE_PROMPT_TEMPLATE = """\
You are a senior quality-engineering lead evaluating how well an AI assistant responded to a QA prompt.

## Original Prompt
{prompt}

## LLM Response
{response}

## Task
Score the response on each dimension from 0 to 10.
Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "relevance":    <0-10>,
  "completeness": <0-10>,
  "clarity":      <0-10>,
  "format":       <0-10>,
  "actionable":   <0-10>,
  "observations": "<1-2 sentence plain-English summary of what was good or bad>"
}}
"""

FIX_PROMPT_TEMPLATE = """\
You are a senior prompt engineering expert specializing in quality-engineering prompts.

The prompt below was tested on {llm_names} and received the following scores and observations:

{score_summary}

## Original Prompt
{original_prompt}

## Task
Rewrite the prompt to fix the observed issues so it works well on ALL tested LLMs.
Return ONLY the improved prompt text — no explanation, no preamble, no markdown code fence.
"""


@dataclass
class DimensionScores:
    relevance: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    format: float = 0.0
    actionable: float = 0.0
    observations: str = ""

    @property
    def average(self) -> float:
        vals = [self.relevance, self.completeness, self.clarity, self.format, self.actionable]
        return round(sum(vals) / len(vals), 1)

    @property
    def verdict(self) -> str:
        avg = self.average
        if avg >= PASS_THRESHOLD:
            return "PASS"
        if avg >= PARTIAL_THRESHOLD:
            return "PARTIAL"
        return "FAIL"


@dataclass
class EvalResult:
    llm: str
    scores: DimensionScores
    response: str
    latency_ms: int
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class PromptReport:
    prompt_id: str
    prompt_title: str
    prompt_text: str
    results: List[EvalResult] = field(default_factory=list)
    fixed_prompt: Optional[str] = None
    overall_verdict: str = "UNKNOWN"   # PASS | PARTIAL | FAIL

    def needs_fix(self) -> bool:
        active = [r for r in self.results if not r.skipped]
        return any(r.scores.verdict in ("FAIL", "PARTIAL") for r in active)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _build_judge():
    """Return a cheap Claude adapter for judging responses."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        from adapters.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key=key, model="claude-haiku-4-5-20251001", timeout=30)
    except Exception:
        return None


def _parse_scores(raw: str) -> DimensionScores:
    """Extract JSON scores from the judge response; fall back to zeros on parse error."""
    try:
        # Strip any accidental markdown fences
        clean = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(clean)
        return DimensionScores(
            relevance=float(data.get("relevance", 0)),
            completeness=float(data.get("completeness", 0)),
            clarity=float(data.get("clarity", 0)),
            format=float(data.get("format", 0)),
            actionable=float(data.get("actionable", 0)),
            observations=str(data.get("observations", "")),
        )
    except Exception:
        return DimensionScores(observations=f"Could not parse judge response: {raw[:200]}")


def score_response(judge, prompt_text: str, response: str) -> DimensionScores:
    if not response or not response.strip():
        return DimensionScores(observations="No response received from this LLM.")
    judge_prompt = SCORE_PROMPT_TEMPLATE.format(prompt=prompt_text, response=response)
    raw = judge.generate_sync(judge_prompt, max_tokens=300, temperature=0.1)
    return _parse_scores(raw)


def generate_fix(judge, original_prompt: str, results: List[EvalResult]) -> str:
    """Ask the judge to produce a better version of the prompt."""
    llm_names = ", ".join(r.llm for r in results if not r.skipped)
    lines = []
    for r in results:
        if r.skipped:
            continue
        lines.append(
            f"**{r.llm}** (avg {r.scores.average}/10, {r.scores.verdict}): "
            f"{r.scores.observations}"
        )
    score_summary = "\n".join(lines)
    fix_prompt = FIX_PROMPT_TEMPLATE.format(
        llm_names=llm_names,
        score_summary=score_summary,
        original_prompt=original_prompt,
    )
    return judge.generate_sync(fix_prompt, max_tokens=1000, temperature=0.3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    prompt_id: str,
    prompt_title: str,
    prompt_text: str,
    llm_results: List[LLMResult],
    auto_fix: bool = True,
) -> PromptReport:
    """
    Score all LLM results, compute verdicts, and optionally generate a fixed prompt.

    Args:
        prompt_id:    Prompt identifier from the library.
        prompt_title: Human-readable title.
        prompt_text:  The original prompt text.
        llm_results:  Raw responses from runner.run_all_sync().
        auto_fix:     When True and any LLM scores FAIL/PARTIAL, generate a fix.

    Returns:
        PromptReport with scores, verdicts, and (optionally) a fixed prompt.
    """
    judge = _build_judge()
    eval_results: List[EvalResult] = []

    for lr in llm_results:
        if lr.error:
            eval_results.append(EvalResult(
                llm=lr.llm,
                scores=DimensionScores(),
                response="",
                latency_ms=0,
                skipped=True,
                skip_reason=lr.error,
            ))
            continue

        scores = score_response(judge, prompt_text, lr.response) if judge else DimensionScores(
            observations="ANTHROPIC_API_KEY not set — scoring skipped."
        )
        eval_results.append(EvalResult(
            llm=lr.llm,
            scores=scores,
            response=lr.response,
            latency_ms=lr.latency_ms,
        ))

    # Overall verdict = worst across active LLMs
    active = [r for r in eval_results if not r.skipped]
    if not active:
        overall = "SKIPPED"
    elif all(r.scores.verdict == "PASS" for r in active):
        overall = "PASS"
    elif any(r.scores.verdict == "FAIL" for r in active):
        overall = "FAIL"
    else:
        overall = "PARTIAL"

    report = PromptReport(
        prompt_id=prompt_id,
        prompt_title=prompt_title,
        prompt_text=prompt_text,
        results=eval_results,
        overall_verdict=overall,
    )

    if auto_fix and report.needs_fix() and judge:
        report.fixed_prompt = generate_fix(judge, prompt_text, eval_results)

    return report
