"""
evals/structured_evals.py: Metrics for structured outputs, prompt adherence, and G-Eval.

json_correctness  — output is valid JSON and contains the expected keys.
prompt_alignment  — output follows the constraints in the system prompt.
GEvalMetric       — user-defined chain-of-thought evaluation; wraps as a registered metric.
"""

import json
import re
from typing import List, Optional, Dict

from core.metric_registry import register_metric
from evals.llm_judge import get_llm_client, _extract_score_from_response, _JUDGE_SCORE_SCALE


# ============================================================================
# JSON CORRECTNESS
# ============================================================================

@register_metric(
    "json_correctness",
    description="Output is valid JSON with all expected keys present",
    tags=["deterministic", "structured", "extraction"],
    capabilities=["extraction", "generation"],
)
def json_correctness(test_case, actual_output):
    """
    Two sub-scores:
      json_valid             — 1.0 if the output parses as JSON, 0.0 otherwise.
      json_key_completeness  — fraction of expected keys present (1.0 when none required).
      json_correctness       — harmonic mean of the two sub-scores.

    Required keys are taken from test_case.expected_json (if set).
    JSON may be embedded in a markdown code fence and will still be extracted.
    """
    text = actual_output.strip()
    parsed = _try_parse_json(text)
    json_valid = 1.0 if parsed is not None else 0.0

    required_keys: List[str] = []
    if hasattr(test_case, "expected_json") and test_case.expected_json:
        required_keys = list(test_case.expected_json.keys())

    if json_valid == 0.0:
        return {
            "json_correctness": 0.0,
            "json_valid": 0.0,
            "json_key_completeness": 0.0,
        }

    if required_keys and isinstance(parsed, dict):
        present = sum(1 for k in required_keys if k in parsed)
        key_completeness = present / len(required_keys)
    else:
        key_completeness = 1.0

    correctness = (json_valid + key_completeness) / 2
    return {
        "json_correctness": round(correctness, 4),
        "json_valid": json_valid,
        "json_key_completeness": round(key_completeness, 4),
    }


def _try_parse_json(text: str):
    """Try to parse JSON; also extracts JSON from markdown code fences."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to extract from ```json ... ``` or ``` ... ``` fences
    match = re.search(r'```(?:json)?\s*(\{.+?\}|\[.+?\])\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ============================================================================
# PROMPT ALIGNMENT
# ============================================================================

_PROMPT_ALIGNMENT_TEMPLATE = """You are evaluating whether an AI assistant's response follows the instructions given in its system prompt.

System Prompt (the instructions the AI must follow):
{system_prompt}

AI Response:
{output}

Does the response comply with ALL instructions in the system prompt?
Consider: required tone, format restrictions, forbidden topics, persona, language rules.

Score 1-5:
1 = Response clearly violates the system prompt
2 = Response ignores several key instructions
3 = Response partially follows the system prompt
4 = Response mostly follows instructions with minor deviations
5 = Response is perfectly aligned with all system prompt requirements

Format: REASONING: <your reasoning> SCORE: <1-5>"""


@register_metric(
    "prompt_alignment",
    description="How well the output follows the system prompt's constraints",
    tags=["judge", "alignment"],
    capabilities=["generation"],
)
def prompt_alignment(test_case, actual_output):
    """Score whether the output obeys the test_case.system_prompt instructions."""
    system_prompt = getattr(test_case, "system_prompt", None)
    if not system_prompt:
        return {"prompt_alignment": 1.0}

    client = get_llm_client()
    if client is None:
        return {"prompt_alignment": _prompt_alignment_heuristic(system_prompt, actual_output)}
    prompt = _PROMPT_ALIGNMENT_TEMPLATE.format(system_prompt=system_prompt, output=actual_output)
    try:
        response = client.generate(prompt)
        score = _extract_score_from_response(response) / _JUDGE_SCORE_SCALE
        return {"prompt_alignment": round(score, 4)}
    except Exception:
        return {"prompt_alignment": _prompt_alignment_heuristic(system_prompt, actual_output)}


_FORBIDDEN_MARKERS = [
    "do not", "never", "must not", "avoid", "prohibited", "forbidden",
    "do not say", "don't say", "refrain from",
]


def _prompt_alignment_heuristic(system_prompt: str, output: str) -> float:
    """
    Scan for forbidden-phrase patterns in the system prompt and check if the
    output violates them. Returns a score between 0.0 and 1.0.
    """
    system_lower = system_prompt.lower()
    output_lower  = output.lower()
    violations = 0
    for marker in _FORBIDDEN_MARKERS:
        idx = system_lower.find(marker)
        while idx != -1:
            # Extract a few words after the forbidden-phrase marker as the forbidden content
            after = system_lower[idx + len(marker):idx + len(marker) + 40].strip()
            fragment = " ".join(after.split()[:3])
            if fragment and fragment in output_lower:
                violations += 1
            idx = system_lower.find(marker, idx + 1)
    if violations > 0:
        return max(0.0, round(1.0 - violations * 0.25, 4))
    return 0.8  # Assume mostly aligned when no explicit violations detected


# ============================================================================
# G-EVAL (CHAIN-OF-THOUGHT EVALUATION)
# ============================================================================

_GEVAL_TEMPLATE = """Task Description: {rubric_description}

Evaluation Steps:
{steps}

Input: {input}
{context_block}Response to Evaluate: {output}

Work through each evaluation step above, noting your observations.
Then give an overall score from 1 to 5 reflecting all the steps.

Format:
STEP EVALUATIONS: <brief notes on each step>
SCORE: <1-5>"""


class GEvalMetric:
    """
    G-Eval: a flexible LLM-based evaluator where YOU define the evaluation logic as steps.

    Unlike fixed rubrics (correctness, fluency), G-Eval lets you write plain-English
    evaluation steps that mirror how a domain expert would review a response.

    Example usage:
        evaluator = GEvalMetric(
            name="insurance_response_quality",
            rubric_description="Evaluate if a chatbot correctly handles an insurance claim query",
            evaluation_steps=[
                "Check if the response acknowledges the customer's concern empathetically.",
                "Check if the response provides accurate information about the claims process.",
                "Check if the response avoids making unauthorised commitments or guarantees.",
                "Check if the response directs the customer to appropriate next steps.",
            ],
        )
        score = evaluator.evaluate(test_case, actual_output)   # returns 0.0–1.0

        # Optionally register as a named metric in the MetricRegistry:
        evaluator.as_metric()
    """

    def __init__(
        self,
        name: str,
        rubric_description: str,
        evaluation_steps: List[str],
    ) -> None:
        self.name = name
        self.rubric_description = rubric_description
        self.evaluation_steps = evaluation_steps

    def evaluate(self, test_case, actual_output: str) -> float:
        """Run G-Eval and return a normalised score in [0.0, 1.0]."""
        client = get_llm_client()
        if client is None:
            return self._heuristic_score(test_case, actual_output)
        prompt = self._build_prompt(test_case, actual_output)
        try:
            response = client.generate(prompt)
            return round(_extract_score_from_response(response) / _JUDGE_SCORE_SCALE, 4)
        except Exception:
            return self._heuristic_score(test_case, actual_output)

    def _build_prompt(self, test_case, output: str) -> str:
        steps_text = "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(self.evaluation_steps)
        )
        context_block = f"Context: {test_case.context}\n" if getattr(test_case, "context", None) else ""
        return _GEVAL_TEMPLATE.format(
            rubric_description=self.rubric_description,
            steps=steps_text,
            input=test_case.input,
            context_block=context_block,
            output=output,
        )

    def _heuristic_score(self, test_case, actual_output: str) -> float:
        """Fallback: keyword overlap + length proxy for a non-zero heuristic."""
        if not actual_output.strip():
            return 0.0
        input_words  = set(test_case.input.lower().split())
        output_words = set(actual_output.lower().split())
        overlap = len(input_words & output_words) / max(len(input_words), 1)
        length_score = min(1.0, len(actual_output.split()) / 30)
        return round((overlap + length_score) / 2, 4)

    def as_metric(self):
        """Register this GEvalMetric instance as a named metric in the MetricRegistry."""
        evaluator = self

        @register_metric(
            evaluator.name,
            description=f"G-Eval: {evaluator.rubric_description[:80]}",
            tags=["judge", "g-eval"],
            capabilities=["generation"],
        )
        def _metric_fn(test_case, actual_output):
            score = evaluator.evaluate(test_case, actual_output)
            return {evaluator.name: score}

        return _metric_fn
