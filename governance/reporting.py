# Reporting Guardrails - Category 7
# Reports that (a) separate raw results from interpreted conclusions, (b) always show
# failure rates next to success rates, (c) break out subgroup performance, (d) cannot hide
# missing data or cherry-picked subsets, and (e) stamp every run with the versions that
# produced it. This complements reports/* (which render results); here we govern *what*
# a report is allowed to omit.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from governance.metric_spec import bootstrap_ci


# ============================================================================
# RUN MANIFEST  (every report carries its provenance)
# ============================================================================

@dataclass
class RunManifest:
    """The version stamps that make a result reproducible: required on every report."""
    run_id: str
    model: str
    dataset_version: str
    prompt_version: str = "n/a"
    judge_version: str = "n/a"
    framework_version: str = "n/a"
    objective: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ============================================================================
# GOVERNED REPORT
# ============================================================================

def build_governance_report(
    manifest: RunManifest,
    results: List[Dict[str, Any]],
    meta: Optional[Dict[str, Dict[str, Any]]] = None,
    subgroup_keys: tuple = ("severity", "capability"),
    expected_total: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Assemble a governed report dict with raw and interpreted sections kept separate.

    Args:
        manifest: version provenance for the run.
        results: list of EvalResult dicts (test_id, passed, scores, notes).
        meta: optional per-test metadata (test_id -> {severity, capability, ...}) for subgroups.
        subgroup_keys: metadata keys to break performance down by.
        expected_total: how many samples *should* be present: if results is fewer, the
            report flags the shortfall instead of silently reporting on a subset.

    Returns:
        A dict with "manifest", "raw", "subgroups", "interpretation", and "integrity".
    """
    meta = meta or {}
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = total - passed
    # Missing/malformed outputs are surfaced, never dropped.
    missing = sum(1 for r in results if any("missing" in n.lower() or "malformed" in n.lower()
                                            for n in r.get("notes", [])))

    pass_scores = [1.0 if r.get("passed") else 0.0 for r in results]
    ci = bootstrap_ci(pass_scores) if pass_scores else {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}

    # ── RAW (uninterpreted facts) ─────────────────────────────────────────
    raw = {
        "total": total,
        "passed": passed,
        "failed": failed,                       # failure rate always shown next to success
        "success_rate": round(passed / total, 4) if total else 0.0,
        "failure_rate": round(failed / total, 4) if total else 0.0,
        "missing_or_malformed": missing,
    }

    # ── SUBGROUPS (no hiding underperforming slices) ──────────────────────
    subgroups: Dict[str, Dict[str, Any]] = {}
    for key in subgroup_keys:
        groups: Dict[str, List[bool]] = {}
        for r in results:
            g = str(meta.get(r.get("test_id"), {}).get(key, "unknown"))
            groups.setdefault(g, []).append(bool(r.get("passed")))
        subgroups[key] = {
            g: {"n": len(v), "pass_rate": round(sum(v) / len(v), 4) if v else 0.0}
            for g, v in groups.items()
        }

    # ── INTEGRITY (cherry-pick / missing-data guard) ──────────────────────
    integrity = {
        "expected_total": expected_total,
        "reported_total": total,
        "complete": expected_total is None or total >= expected_total,
        "omitted": max(0, (expected_total or total) - total),
    }

    # ── INTERPRETATION (clearly separated from raw) ───────────────────────
    interpretation = {
        "verdict": "PASS" if failed == 0 else "FAIL",
        "pass_rate_ci": ci,                      # error bars, not a bare number
        "caveats": _caveats(raw, integrity, subgroups),
        "note": "Interpretation reflects measurement under this manifest; it is not a "
                "statement of absolute model quality.",
    }

    return {
        "manifest": manifest.as_dict(),
        "raw": raw,
        "subgroups": subgroups,
        "integrity": integrity,
        "interpretation": interpretation,
    }


def _caveats(raw: Dict[str, Any], integrity: Dict[str, Any], subgroups: Dict[str, Any]) -> List[str]:
    """Generate honest caveats so conclusions can't oversell the data."""
    caveats = []
    if raw["total"] < 30:
        caveats.append(f"Small sample (n={raw['total']}): results are low-confidence.")
    if raw["missing_or_malformed"] > 0:
        caveats.append(f"{raw['missing_or_malformed']} sample(s) had missing/malformed output.")
    if not integrity["complete"]:
        caveats.append(f"{integrity['omitted']} expected sample(s) are missing from this report.")
    for key, groups in subgroups.items():
        weak = [g for g, v in groups.items() if v["pass_rate"] < 0.5 and g != "unknown"]
        if weak:
            caveats.append(f"Underperforming {key}: {', '.join(weak)}.")
    return caveats


def render_markdown(report: Dict[str, Any]) -> str:
    """Render a governed report to Markdown with raw and interpreted sections separated."""
    m, raw, interp = report["manifest"], report["raw"], report["interpretation"]
    lines = [
        f"# Governed Evaluation Report: {m['run_id']}",
        "",
        f"_Model `{m['model']}` , dataset `{m['dataset_version'][:12]}` , "
        f"judge `{m['judge_version']}` , {m['timestamp']}_",
        "",
        "## Raw results (uninterpreted)",
        f"- Total: **{raw['total']}** , Passed: **{raw['passed']}** , Failed: **{raw['failed']}**",
        f"- Success rate: **{raw['success_rate']:.1%}** , Failure rate: **{raw['failure_rate']:.1%}**",
        f"- Missing/malformed: **{raw['missing_or_malformed']}**",
        "",
        "## Interpretation (separate from raw)",
        f"- Verdict: **{interp['verdict']}**",
        f"- Pass-rate 95% CI: **[{interp['pass_rate_ci']['lower']:.2f}, {interp['pass_rate_ci']['upper']:.2f}]**",
    ]
    if interp["caveats"]:
        lines.append("- Caveats:")
        lines.extend(f"  - {c}" for c in interp["caveats"])
    lines.append(f"\n> {interp['note']}")
    return "\n".join(lines)
