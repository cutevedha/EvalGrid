# Robustness Checks - Category 6
# Detects prompt injection hidden in EVAL INPUTS (not just target inputs), verifies the
# suite actually contains adversarial samples, measures stability across random seeds and
# sample reorderings, and stress-tests edge / long-context cases. Reuses the existing
# guards and synthetic generators instead of reimplementing them.

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Sequence

from guards.prompt_injection import is_prompt_injection
from synthetic.redteam import ATTACK_CATEGORIES
from synthetic.augmentation import generate_edge_cases


# ============================================================================
# PROMPT INJECTION INTO EVAL INPUTS
# ============================================================================

def scan_eval_inputs_for_injection(samples: List[Dict[str, Any]]) -> List[str]:
    """
    Flag eval samples whose *input* contains prompt-injection patterns.

    An attacker who can seed the eval set could smuggle instructions that hijack the judge
    or the harness. This reuses guards/prompt_injection to flag such samples for review.
    Returns the list of flagged sample IDs.
    """
    flagged = []
    for idx, sample in enumerate(samples):
        text = sample.get("input", "")
        if is_prompt_injection(text):
            flagged.append(str(sample.get("id", f"idx{idx}")))
    return flagged


# ============================================================================
# ADVERSARIAL COVERAGE
# ============================================================================

# Risk tags that mark a sample as adversarial (sourced from the red-team categories).
_ADVERSARIAL_TAGS = set(ATTACK_CATEGORIES) | {"adversarial"}


def adversarial_coverage(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Report how much of a suite is adversarial — the spec wants adversarial samples present.

    Looks at each sample's risk_tags against the known red-team categories. Does not
    generate anything (synthetic/redteam owns generation); it only audits coverage.
    """
    adversarial_ids = []
    covered_categories = set()
    for idx, sample in enumerate(samples):
        tags = set(sample.get("risk_tags", []))
        hit = tags & _ADVERSARIAL_TAGS
        if hit:
            adversarial_ids.append(str(sample.get("id", f"idx{idx}")))
            covered_categories |= (tags & set(ATTACK_CATEGORIES))
    total = len(samples)
    return {
        "total_samples": total,
        "adversarial_samples": len(adversarial_ids),
        "adversarial_fraction": round(len(adversarial_ids) / total, 3) if total else 0.0,
        "categories_covered": sorted(covered_categories),
        "categories_missing": sorted(set(ATTACK_CATEGORIES) - covered_categories),
        "has_adversarial": bool(adversarial_ids),
    }


# ============================================================================
# STOCHASTICITY: MULTI-SEED STABILITY
# ============================================================================

def multi_seed_stability(run_fn: Callable[[int], float], seeds: Sequence[int]) -> Dict[str, float]:
    """
    Run a stochastic evaluation across multiple seeds and report stability.

    ``run_fn(seed)`` returns a scalar score for that seed. A wide spread means the result
    is seed-sensitive and a single-seed number would be misleading.
    """
    scores = [float(run_fn(s)) for s in seeds]
    if not scores:
        return {"runs": 0, "mean": 0.0, "stdev": 0.0, "spread": 0.0, "stable": True}
    spread = max(scores) - min(scores)
    return {
        "runs": len(scores),
        "mean": round(mean(scores), 4),
        "stdev": round(pstdev(scores) if len(scores) > 1 else 0.0, 4),
        "spread": round(spread, 4),
        "stable": spread < 0.1,   # <10% swing across seeds
        "scores": [round(s, 4) for s in scores],
    }


# ============================================================================
# METRIC STABILITY ACROSS REORDERINGS
# ============================================================================

def reordering_stability(aggregate_fn: Callable[[List[Any]], float], samples: List[Any],
                         shuffles: int = 5, seed: int = 0) -> Dict[str, float]:
    """
    Check that an aggregate metric does not depend on sample order.

    A correctly implemented aggregate (mean, pass-rate…) must be order-invariant; if the
    value moves when samples are shuffled, the aggregation has a hidden order dependency.
    """
    import random
    rng = random.Random(seed)
    baseline = aggregate_fn(list(samples))
    values = [baseline]
    for _ in range(shuffles):
        shuffled = list(samples)
        rng.shuffle(shuffled)
        values.append(aggregate_fn(shuffled))
    spread = max(values) - min(values)
    return {
        "baseline": round(baseline, 6),
        "max_deviation": round(spread, 6),
        "order_invariant": spread < 1e-9,
    }


# ============================================================================
# EDGE / LONG-CONTEXT STRESS
# ============================================================================

def stress_inputs(base_text: str, long_context_tokens: int = 4000) -> List[str]:
    """
    Build edge-case and long-context variants of an input for stress testing.

    Reuses synthetic/augmentation for the standard edge cases and appends a long-context
    variant to probe context-length handling.
    """
    cases = list(generate_edge_cases(base_text))
    filler = ("lorem ipsum " * long_context_tokens)
    cases.append(f"{filler}\n\n{base_text}")   # long-context variant
    return cases
