# Red Flags - Category 11
# A meta-checker that scans a completed run for the spec's hard "do not ship" signals.
# It does not compute anything itself; it inspects evidence produced by the other
# governance modules and raises a flag when an integrity-violating condition is present.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


# ============================================================================
# RED FLAG MODEL
# ============================================================================

@dataclass
class RedFlag:
    code: str
    description: str
    severity: str = "critical"


# Canonical red flags from the spec, keyed by code.
_RED_FLAG_TEXT = {
    "prompt_tampering": "The framework modified the target's prompts/outputs (it must only measure).",
    "judge_leakage": "The judge prompt contained hidden hints or leaked reference answers.",
    "silent_version_change": "Results changed without a version bump (content hash mismatch).",
    "no_error_bars": "Success rates were reported without error bars or failure examples.",
    "irreproducible_score": "Scoring is not reproducible — a reviewer cannot reproduce the final score.",
}


# ============================================================================
# SCANNER
# ============================================================================

def scan_red_flags(context: Dict[str, Any]) -> List[RedFlag]:
    """
    Inspect a run's evidence and return any triggered red flags.

    Expected context keys (all optional; absent = treated as safe where sensible):
        scope_violations        int  — times the target output changed after capture
        prompts_modified        bool — framework altered target prompts/outputs
        judge_reference_leaks   int  — judge prompts found leaking the gold answer
        version_mismatch        bool — recomputed hash != recorded version id
        report_has_error_bars   bool — report includes CIs / uncertainty
        report_has_failures     bool — report includes concrete failure examples
        scoring_reproducible    bool — identical inputs reproduce identical scores
    """
    flags: List[RedFlag] = []

    if context.get("prompts_modified") or context.get("scope_violations", 0) > 0:
        flags.append(RedFlag("prompt_tampering", _RED_FLAG_TEXT["prompt_tampering"]))

    if context.get("judge_reference_leaks", 0) > 0:
        flags.append(RedFlag("judge_leakage", _RED_FLAG_TEXT["judge_leakage"]))

    if context.get("version_mismatch"):
        flags.append(RedFlag("silent_version_change", _RED_FLAG_TEXT["silent_version_change"]))

    if not context.get("report_has_error_bars", True) or not context.get("report_has_failures", True):
        flags.append(RedFlag("no_error_bars", _RED_FLAG_TEXT["no_error_bars"], severity="high"))

    if not context.get("scoring_reproducible", True):
        flags.append(RedFlag("irreproducible_score", _RED_FLAG_TEXT["irreproducible_score"]))

    return flags


def is_clean(flags: List[RedFlag]) -> bool:
    """True if no critical red flag is present (high-severity flags warn but don't block)."""
    return not any(f.severity == "critical" for f in flags)
