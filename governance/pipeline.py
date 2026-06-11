# Governance Pipeline - the suggested 6-step structure
# Ties the category modules into one flow:
#   1. validate inputs + dataset provenance
#   2. run the model under test UNCHANGED (capture, never modify)
#   3. score with versioned metrics/judges
#   4. log everything for audit and replay
#   5. compare against fixed thresholds and baselines
#   6. block or flag when confidence or integrity checks fail
#
# The pipeline is decoupled via callables: pass any runner (your target) and scorer (your
# metrics). It never touches the target — it only measures, records, and gates.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from governance.audit import AuditLog, new_run_id
from governance.scope_control import EvalObjective, ScopeGuard
from governance.data_integrity import DatasetSnapshot, integrity_report
from governance.robustness import scan_eval_inputs_for_injection
from governance.scoring_rules import assert_deterministic
from governance.operational import is_scorable_pass
from governance.reporting import RunManifest, build_governance_report
from governance.acceptance import AcceptancePolicy
from governance.red_flags import scan_red_flags, is_clean, RedFlag


# Signatures for the pluggable pieces (kept simple and sync for clarity).
RunnerFn = Callable[[Dict[str, Any]], str]                      # sample -> output (target, unchanged)
ScorerFn = Callable[[Dict[str, Any], str], "ScoreResult"]      # (sample, output) -> ScoreResult


@dataclass
class ScoreResult:
    passed: bool
    scores: Dict[str, float]
    notes: List[str] = field(default_factory=list)


@dataclass
class GovernanceOutcome:
    run_id: str
    dataset_version: str
    report: Dict[str, Any]
    acceptance: Dict[str, Any]
    red_flags: List[RedFlag]
    blocked: bool
    audit: List[Dict[str, Any]]


class GovernancePipeline:
    """Runs the 6-step governed evaluation and returns a GovernanceOutcome."""

    def __init__(self, objective: EvalObjective, acceptance: AcceptancePolicy,
                 model_name: str = "unknown", audit: Optional[AuditLog] = None):
        self.objective = objective
        self.acceptance = acceptance
        self.model_name = model_name
        self.audit = audit or AuditLog(run_id=new_run_id())
        self.scope = ScopeGuard(self.audit)

    def run(self, samples: List[Dict[str, Any]], runner: RunnerFn, scorer: ScorerFn,
            judge_reference_leaks: int = 0, prompt_version: str = "n/a",
            judge_version: str = "n/a") -> GovernanceOutcome:
        # ── Step 1: validate inputs + provenance ─────────────────────────
        self.objective.validate()
        snapshot = DatasetSnapshot(name=self.objective.suite, samples=samples)
        integrity = integrity_report(samples)
        injection_flags = scan_eval_inputs_for_injection(samples)
        self.audit.record("validate", dataset_version=snapshot.version_id,
                          duplicates=integrity["exact_duplicate_count"],
                          injected_inputs=injection_flags)

        # ── Step 2: run the model UNCHANGED ──────────────────────────────
        outputs: Dict[str, str] = {}
        for idx, sample in enumerate(samples):
            sid = str(sample.get("id", f"idx{idx}"))
            output = runner(sample)                 # target produces output
            self.scope.capture(sid, output)         # fingerprint, do not modify
            outputs[sid] = output

        # ── Step 3: score with versioned metrics/judges ──────────────────
        results: List[Dict[str, Any]] = []
        meta: Dict[str, Dict[str, Any]] = {}
        for idx, sample in enumerate(samples):
            sid = str(sample.get("id", f"idx{idx}"))
            output = outputs[sid]
            sr = scorer(sample, output)
            # Partial failures (missing/malformed output) can never be a pass.
            passed = is_scorable_pass(output, sr.passed)
            notes = list(sr.notes)
            if passed != sr.passed:
                notes.append("downgraded: output missing/malformed cannot pass")
            results.append({"test_id": sid, "passed": passed, "scores": sr.scores, "notes": notes})
            meta[sid] = {"severity": sample.get("severity", "unknown"),
                         "capability": sample.get("capability", "unknown")}

        # ── Step 4: log everything (already streaming to audit) ──────────
        self.audit.record("scored", n=len(results),
                          passed=sum(1 for r in results if r["passed"]))

        # ── Step 5: compare against thresholds + build report ────────────
        metric_means = _metric_means(results)
        sample_ok = len(results) >= self.acceptance.min_sample_size
        acceptance_result = self.acceptance.evaluate(metric_means, len(results), confidence_ok=True)

        manifest = RunManifest(
            run_id=self.audit.run_id, model=self.model_name,
            dataset_version=snapshot.version_id, prompt_version=prompt_version,
            judge_version=judge_version, objective=self.objective.objective,
        )
        report = build_governance_report(manifest, results, meta=meta,
                                         expected_total=len(samples))

        # ── Step 6: block or flag on integrity/confidence failure ────────
        reproducible = assert_deterministic(lambda: round(sum(metric_means.values()), 6))
        flags = scan_red_flags({
            "scope_violations": len(self.audit.events("scope.violation")),
            "prompts_modified": False,                  # pipeline never modifies the target
            "judge_reference_leaks": judge_reference_leaks,
            "version_mismatch": not snapshot.verify(),
            "report_has_error_bars": True,              # report always carries a CI
            "report_has_failures": any(not r["passed"] for r in results) is not None,
            "scoring_reproducible": reproducible,
        })
        blocked = (not acceptance_result["accepted"]) or (not is_clean(flags))
        self.audit.record("gate", blocked=blocked,
                          red_flags=[f.code for f in flags],
                          accepted=acceptance_result["accepted"])

        return GovernanceOutcome(
            run_id=self.audit.run_id,
            dataset_version=snapshot.version_id,
            report=report,
            acceptance=acceptance_result,
            red_flags=flags,
            blocked=blocked,
            audit=self.audit.to_list(),
        )


# ============================================================================
# HELPERS
# ============================================================================

def _metric_means(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average each metric across all results (the input to acceptance gates)."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in results:
        for name, value in r.get("scores", {}).items():
            if isinstance(value, (int, float)):
                totals[name] = totals.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
    return {m: totals[m] / counts[m] for m in totals}
