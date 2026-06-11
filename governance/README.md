# EvalGrid Governance Layer

A non-blocking guardrail layer that wraps the evaluation engine with integrity,
traceability, and acceptance controls.

**Design invariant:** governance **measures and records**: it never modifies the target
model and never throttles the core eval. Every operational control (timeouts, cost
ceilings, rate limits) is opt-in and a no-op unless explicitly configured.

## Category -> module map

| # | Governance category | Module | Key pieces |
|---|---------------------|--------|-----------|
| 1 | Scope control | [`scope_control.py`](scope_control.py) + [`audit.py`](audit.py) | `EvalObjective`, `ScopeGuard` (immutable output fingerprint), `ReversibleTransform`, `AuditLog` (run IDs) |
| 2 | Data integrity | [`data_integrity.py`](data_integrity.py) | `DatasetSnapshot` (content-hash versions), `detect_split_leakage`, `find_duplicates`/`find_near_duplicates`, `detect_contamination`, provenance |
| 3 | Metric validity | [`metric_spec.py`](metric_spec.py) | `MetricSpec` (definition + rationale), `MetricCatalog.coverage`, `bootstrap_ci`, `cohen_kappa` |
| 4 | Scoring guardrails | [`scoring_rules.py`](scoring_rules.py) | `classify_output` (missing/malformed/truncated), `ScoringPolicy`, `assert_deterministic`, `break_tie`, `detect_outliers`, `aggregate` |
| 5 | Judge quality | [`judge_quality.py`](judge_quality.py) | `JudgePromptRegistry` (versioned), `has_reference_leak`, `position_consistency`, `length_bias_correlation`, `disagreement`, `HumanOverride` |
| 6 | Robustness | [`robustness.py`](robustness.py) | `scan_eval_inputs_for_injection`, `adversarial_coverage`, `multi_seed_stability`, `reordering_stability`, `stress_inputs` |
| 7 | Reporting | [`reporting.py`](reporting.py) | `RunManifest` (version stamps), `build_governance_report` (raw vs interpreted, failure rates, subgroups, missing-data integrity) |
| 8 | Operational safety | [`operational.py`](operational.py) | `ResiliencePolicy`/`run_resilient`, `CostMeter`, `RateLimiter`, `is_scorable_pass`, `AccessControl` |
| 9 | Change management | [`change_management.py`](change_management.py) | `ComponentVersions`, `requires_revalidation`, `BaselineStore` (+ rollback), `mark_breaking_changes` |
| 10 | Acceptance criteria | [`acceptance.py`](acceptance.py) | `AcceptancePolicy` (min sample size, critical vs exploratory gates, no single-metric release, measurement-confidence vs model-quality) |
| 11 | Red flags | [`red_flags.py`](red_flags.py) | `scan_red_flags`, `is_clean` |
|: | Suggested 6-step structure | [`pipeline.py`](pipeline.py) | `GovernancePipeline`: validate -> run-unchanged -> score -> log -> compare -> gate |

## Reuse, not duplication

The layer builds on existing framework parts instead of re-implementing them:
- **Adversarial generation** -> reuses `synthetic/redteam.py` + the autonomous agent.
- **Injection detection** -> reuses `guards/prompt_injection.py`.
- **Edge cases** -> reuses `synthetic/augmentation.py`.
- **Judge rubrics** -> versions the existing `evals/llm_judge.py` templates.
- **Per-metric averaging gates** -> complements `pipelines/GatingRunner` with a policy layer.

## Quick start

```python
from governance import (
    GovernancePipeline, EvalObjective, AcceptancePolicy, ScoreResult,
    TIER_CRITICAL, TIER_EXPLORATORY,
)

policy = (AcceptancePolicy(min_sample_size=30)
          .add_gate("policy_safe", 1.0, tier=TIER_CRITICAL)
          .add_gate("refused", 1.0, tier=TIER_EXPLORATORY))

pipe = GovernancePipeline(
    EvalObjective("safety", "ensure refusals on unsafe asks"),
    policy, model_name="my-model",
)

# runner = your target (called UNCHANGED); scorer = your metrics
outcome = pipe.run(samples, runner, scorer)

print("BLOCKED" if outcome.blocked else "RELEASE OK")
print(outcome.report["raw"])            # raw facts (success + failure rates)
print(outcome.report["interpretation"]) # interpreted verdict + error bars (kept separate)
print(outcome.red_flags)                # any integrity red flags
print(outcome.audit)                    # full traceable event log
```

## What is intentionally *not* automated

A few spec items require humans or external systems and are provided as **hooks**, not
fake automation:
- **Inter-rater agreement / judge calibration**: `cohen_kappa`, `agreement_rate`, and the
  judge-bias probes are provided, but you supply the human labels.
- **Human override path**: `HumanOverride` records a person's decision; it does not invent one.
- **Access control**: `AccessControl` enforces roles you define; wire it to your real auth.
