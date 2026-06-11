# EvalGrid governance package
# A non-blocking guardrail layer that wraps the evaluation engine with integrity,
# traceability and acceptance controls. It MEASURES and RECORDS — it never modifies the
# target model or throttles the core eval. Each module maps to one category of the
# governance spec; see governance/README.md for the full mapping.

# Category 1 — scope control + shared audit foundation
from governance.audit import AuditLog, AuditEvent, new_run_id, content_hash
from governance.scope_control import EvalObjective, ScopeGuard, ReversibleTransform

# Category 2 — data integrity
from governance.data_integrity import (
    DatasetSnapshot, detect_split_leakage, find_duplicates, find_near_duplicates,
    detect_contamination, integrity_report,
)

# Category 3 — metric validity
from governance.metric_spec import MetricSpec, MetricCatalog, bootstrap_ci, cohen_kappa, agreement_rate

# Category 4 — scoring guardrails
from governance.scoring_rules import (
    ScoringPolicy, classify_output, assert_deterministic, aggregate, detect_outliers,
    break_tie, OUTPUT_OK, OUTPUT_MISSING, OUTPUT_MALFORMED, OUTPUT_TRUNCATED,
)

# Category 5 — judge quality
from governance.judge_quality import (
    JudgePromptRegistry, has_reference_leak, position_consistency, length_bias_correlation,
    style_invariance, disagreement, HumanOverride,
)

# Category 6 — robustness
from governance.robustness import (
    scan_eval_inputs_for_injection, adversarial_coverage, multi_seed_stability,
    reordering_stability, stress_inputs,
)

# Category 7 — reporting
from governance.reporting import RunManifest, build_governance_report, render_markdown

# Category 8 — operational safety
from governance.operational import (
    ResiliencePolicy, run_resilient, CostMeter, RateLimiter, BudgetExceeded,
    is_scorable_pass, AccessControl,
)

# Category 9 — change management
from governance.change_management import (
    ComponentVersions, requires_revalidation, BaselineStore, mark_breaking_changes,
)

# Category 10 — acceptance
from governance.acceptance import AcceptancePolicy, Gate, TIER_CRITICAL, TIER_EXPLORATORY

# Category 11 — red flags
from governance.red_flags import scan_red_flags, is_clean, RedFlag

# Suggested structure — 6-step pipeline
from governance.pipeline import GovernancePipeline, GovernanceOutcome, ScoreResult

__all__ = [
    # foundation + scope
    "AuditLog", "AuditEvent", "new_run_id", "content_hash",
    "EvalObjective", "ScopeGuard", "ReversibleTransform",
    # data integrity
    "DatasetSnapshot", "detect_split_leakage", "find_duplicates", "find_near_duplicates",
    "detect_contamination", "integrity_report",
    # metric validity
    "MetricSpec", "MetricCatalog", "bootstrap_ci", "cohen_kappa", "agreement_rate",
    # scoring
    "ScoringPolicy", "classify_output", "assert_deterministic", "aggregate", "detect_outliers",
    "break_tie", "OUTPUT_OK", "OUTPUT_MISSING", "OUTPUT_MALFORMED", "OUTPUT_TRUNCATED",
    # judge quality
    "JudgePromptRegistry", "has_reference_leak", "position_consistency",
    "length_bias_correlation", "style_invariance", "disagreement", "HumanOverride",
    # robustness
    "scan_eval_inputs_for_injection", "adversarial_coverage", "multi_seed_stability",
    "reordering_stability", "stress_inputs",
    # reporting
    "RunManifest", "build_governance_report", "render_markdown",
    # operational
    "ResiliencePolicy", "run_resilient", "CostMeter", "RateLimiter", "BudgetExceeded",
    "is_scorable_pass", "AccessControl",
    # change management
    "ComponentVersions", "requires_revalidation", "BaselineStore", "mark_breaking_changes",
    # acceptance
    "AcceptancePolicy", "Gate", "TIER_CRITICAL", "TIER_EXPLORATORY",
    # red flags
    "scan_red_flags", "is_clean", "RedFlag",
    # pipeline
    "GovernancePipeline", "GovernanceOutcome", "ScoreResult",
]
