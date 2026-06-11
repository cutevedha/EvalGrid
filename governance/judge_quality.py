# Judge Quality Controls - Category 5
# Versioned + tested LLM-as-judge prompts, bias checks (position / length / style), guards
# against reference-answer leakage into judge prompts, disagreement thresholds that trigger
# review, and a human-override path for high-impact decisions. Builds on evals/llm_judge.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from governance.audit import AuditLog, content_hash
from governance.data_integrity import _normalize


# ============================================================================
# VERSIONED JUDGE PROMPTS
# ============================================================================

@dataclass
class JudgePromptVersion:
    version: str          # content hash
    text: str
    note: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class JudgePromptRegistry:
    """
    Stores judge prompts with content-hash versions and a changelog.

    A judge prompt can never change "silently": editing the text yields a new version id,
    and the full history is retained so any score can be traced to the exact prompt used.
    """

    def __init__(self):
        self._history: Dict[str, List[JudgePromptVersion]] = {}

    def register(self, name: str, text: str, note: str = "") -> str:
        version = content_hash(text)
        versions = self._history.setdefault(name, [])
        if not versions or versions[-1].version != version:
            versions.append(JudgePromptVersion(version=version, text=text, note=note))
        return version

    def current(self, name: str) -> Optional[JudgePromptVersion]:
        versions = self._history.get(name)
        return versions[-1] if versions else None

    def history(self, name: str) -> List[JudgePromptVersion]:
        return list(self._history.get(name, []))

    def seed_from_builtin(self) -> None:
        """Version the framework's built-in judge rubrics so they are tracked from day one."""
        try:
            from evals.llm_judge import JUDGE_TEMPLATES
            for rubric, text in JUDGE_TEMPLATES.items():
                self.register(f"llm_judge:{rubric}", text, note="builtin")
        except Exception:  # noqa: BLE001 - judge module optional
            pass


# ============================================================================
# REFERENCE-ANSWER LEAKAGE PREVENTION
# ============================================================================

def has_reference_leak(judge_prompt: str, reference_answer: str) -> bool:
    """
    True if the gold/reference answer appears inside the judge prompt.

    Leaking the reference into the judge prompt lets the judge "cheat", so this is checked
    before a judge prompt is used. Comparison is normalised to catch whitespace/case tricks.
    """
    if not reference_answer.strip():
        return False
    return _normalize(reference_answer) in _normalize(judge_prompt)


# ============================================================================
# JUDGE BIAS CHECKS  (position / length / style)
# ============================================================================

def position_consistency(pairwise_judge: Callable[[str, str], int], pairs: Sequence[Tuple[str, str]]) -> float:
    """
    Position-bias probe for a pairwise judge.

    ``pairwise_judge(a, b)`` returns 0 if it prefers the first argument, 1 if the second.
    Each pair is judged in both orders; a position-unbiased judge picks the *same content*
    regardless of order. Returns the consistency rate in [0,1] (1.0 = no position bias).
    """
    if not pairs:
        return 1.0
    consistent = 0
    for a, b in pairs:
        first = pairwise_judge(a, b)      # 0 -> a, 1 -> b
        second = pairwise_judge(b, a)     # 0 -> b, 1 -> a
        # Consistent if the same item wins in both orders.
        winner_1 = a if first == 0 else b
        winner_2 = b if second == 0 else a
        if winner_1 == winner_2:
            consistent += 1
    return round(consistent / len(pairs), 4)


def length_bias_correlation(score_fn: Callable[[str], float], texts: Sequence[str]) -> float:
    """
    Pearson correlation between output length and judge score.

    A large positive value suggests the judge rewards verbosity rather than quality.
    Returns 0.0 when it cannot be computed (constant scores or <2 samples).
    """
    if len(texts) < 2:
        return 0.0
    lengths = [float(len(t)) for t in texts]
    scores = [float(score_fn(t)) for t in texts]
    return _pearson(lengths, scores)


def style_invariance(score_fn: Callable[[str], float], variants: Sequence[str]) -> float:
    """
    Score spread across stylistic variants of the *same* content.

    Variants should mean the same thing; a robust judge scores them similarly, so a small
    spread (returned value) is good. 0.0 = perfectly invariant.
    """
    if len(variants) < 2:
        return 0.0
    scores = [float(score_fn(v)) for v in variants]
    return round(max(scores) - min(scores), 4)


# ============================================================================
# DISAGREEMENT + HUMAN OVERRIDE
# ============================================================================

def disagreement(scores: Sequence[float], threshold: float = 0.3) -> bool:
    """True if judges disagree enough (score range >= threshold) to warrant human review."""
    if len(scores) < 2:
        return False
    return (max(scores) - min(scores)) >= threshold


@dataclass
class OverrideDecision:
    sample_id: str
    decision: str          # e.g. "pass" / "fail"
    reviewer: str
    reason: str
    at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class HumanOverride:
    """
    A human override path for high-impact decisions.

    Overrides are recorded (and audited) so the final verdict on a contested sample can be
    set by a person: and that intervention is itself fully traceable.
    """

    def __init__(self, audit: Optional[AuditLog] = None):
        self.audit = audit
        self._overrides: Dict[str, OverrideDecision] = {}

    def set(self, sample_id: str, decision: str, reviewer: str, reason: str = "") -> None:
        record = OverrideDecision(sample_id=sample_id, decision=decision, reviewer=reviewer, reason=reason)
        self._overrides[sample_id] = record
        if self.audit:
            self.audit.record("judge.override", sample_id=sample_id, decision=decision,
                              reviewer=reviewer, reason=reason)

    def get(self, sample_id: str) -> Optional[OverrideDecision]:
        return self._overrides.get(sample_id)


# ============================================================================
# HELPERS
# ============================================================================

def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return round(cov / (vx * vy), 4)
