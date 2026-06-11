# Eval Report - Structured output of an autonomous agent run
# Captures per-probe findings, the round-by-round trace, and an overall verdict so the
# result can be rendered to the existing HTML/JSON/Markdown reporters or inspected directly.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from core.schemas import EvalResult


# ============================================================================
# FINDINGS
# ============================================================================

@dataclass
class ProbeFinding:
    """Aggregated result for a single probe after all rounds it ran in."""
    probe: str                       # Probe name
    capability: str
    severity: str
    cases_run: int                   # Total cases executed across rounds
    pass_rate: float                 # Fraction of cases that passed the gate
    mean_gate_score: float           # Mean of the probe's gate metrics across cases
    weakest_metric: str = ""          # Gate metric with the lowest mean score
    weakest_metric_score: float = 1.0
    failing_inputs: List[str] = field(default_factory=list)  # Sample of inputs that failed

    @property
    def is_weak(self) -> bool:
        """A probe is 'weak' (worth drilling into) when most cases fail."""
        return self.pass_rate < 0.8


@dataclass
class RoundRecord:
    """Snapshot of one adaptive round."""
    round_number: int
    probes_run: List[str]
    cases_run: int
    passed: int
    pass_rate: float


# ============================================================================
# REPORT
# ============================================================================

@dataclass
class EvalReport:
    """The full result of an EvalAgent run."""
    goal: str
    target: str
    started_at: datetime
    finished_at: datetime
    rounds: List[RoundRecord] = field(default_factory=list)
    findings: List[ProbeFinding] = field(default_factory=list)
    results: List[EvalResult] = field(default_factory=list)  # Every EvalResult produced
    summary: str = ""                # Natural-language verdict

    # ------------------------------------------------------------------
    # DERIVED METRICS
    # ------------------------------------------------------------------

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def overall_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def passed(self) -> bool:
        """Overall verdict: no critical/high probe came back weak."""
        return not any(
            f.is_weak and f.severity in ("critical", "high") for f in self.findings
        )

    def weak_findings(self) -> List[ProbeFinding]:
        """Findings sorted worst-first, limited to genuinely weak probes."""
        return sorted(
            (f for f in self.findings if f.is_weak),
            key=lambda f: f.pass_rate,
        )

    # ------------------------------------------------------------------
    # SERIALISATION
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view suitable for JSON export or the existing reporters."""
        return {
            "goal": self.goal,
            "target": self.target,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "verdict": "PASS" if self.passed else "FAIL",
            "overall_pass_rate": round(self.overall_pass_rate, 4),
            "total_cases": self.total_cases,
            "summary": self.summary,
            "rounds": [vars(r) for r in self.rounds],
            "findings": [vars(f) for f in self.findings],
        }

    def result_dicts(self) -> List[Dict[str, Any]]:
        """Every EvalResult as a dict — feed straight into reports.* generators."""
        return [r.model_dump() for r in self.results]
