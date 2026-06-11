# Agent HTML Report - Visualizes an autonomous EvalAgent run
# Unlike the generic results dashboard, this renders the agent's *process*: the
# round-by-round adaptive drilling (how probes narrow each round), per-probe findings,
# and the final verdict. Takes an agent.report.EvalReport and writes a standalone HTML file.

from html import escape
from pathlib import Path


# ============================================================================
# SMALL HELPERS
# ============================================================================

def _pct(value: float) -> str:
    """Format a 0-1 fraction as a whole-number percentage."""
    return f"{value * 100:.0f}%"


def _severity_class(severity: str) -> str:
    """Map a severity label to a CSS class for colour coding."""
    return {
        "critical": "sev-critical",
        "high": "sev-high",
        "medium": "sev-medium",
        "low": "sev-low",
    }.get(severity, "sev-medium")


def _bar(pass_rate: float) -> str:
    """Render a pass-rate as a coloured progress bar (green high, red low)."""
    pct = max(0.0, min(1.0, pass_rate)) * 100
    # Hue 0 (red) -> 120 (green) scaled by the pass rate.
    hue = int(120 * pass_rate)
    return (
        f'<div class="bar"><div class="bar-fill" '
        f'style="width:{pct:.0f}%;background:hsl({hue},65%,45%)"></div>'
        f'<span class="bar-label">{pct:.0f}%</span></div>'
    )


# ============================================================================
# MAIN RENDERER
# ============================================================================

def generate_agent_html_report(report, path: str, title: str = None) -> str:
    """
    Render an agent run to a standalone HTML dashboard.

    Args:
        report: An agent.report.EvalReport instance.
        path: Output file path (e.g. "output/agent_report.html").
        title: Optional page title (defaults to the run goal).

    Returns:
        The path that was written.
    """
    title = title or f"Auto Eval — {report.goal}"
    verdict = "PASS" if report.passed else "FAIL"
    verdict_class = "verdict-pass" if report.passed else "verdict-fail"

    summary_cards = _render_summary_cards(report, verdict, verdict_class)
    rounds_section = _render_rounds(report)
    findings_section = _render_findings(report)
    coverage_section = _render_metric_coverage(report)
    summary_text = escape(report.summary).replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(title)}</title>
{_CSS}
</head>
<body>
<header>
  <div class="verdict-badge {verdict_class}">{verdict}</div>
  <h1>{escape(title)}</h1>
  <p class="meta">
    Target: <b>{escape(report.target)}</b> &nbsp;·&nbsp;
    {report.total_cases} cases &nbsp;·&nbsp;
    {len(report.rounds)} round(s) &nbsp;·&nbsp;
    {escape(report.started_at.strftime('%Y-%m-%d %H:%M:%S'))} UTC
  </p>
</header>
<main>
  {summary_cards}

  <section>
    <h2>Verdict summary</h2>
    <div class="summary-box">{summary_text}</div>
  </section>

  {rounds_section}

  {findings_section}

  {coverage_section}
</main>
<footer>EvalGrid · autonomous agent report</footer>
</body>
</html>"""

    Path(path).write_text(html, encoding="utf-8")
    return path


# ============================================================================
# SECTION RENDERERS
# ============================================================================

def _render_summary_cards(report, verdict: str, verdict_class: str) -> str:
    """Top-of-page metric cards."""
    weak = len(report.weak_findings())
    metrics_per_case = max((len(r.scores) for r in report.results), default=0)
    return f"""
  <div class="cards">
    <div class="card">
      <h3>Verdict</h3>
      <p class="big {verdict_class}">{verdict}</p>
    </div>
    <div class="card">
      <h3>Overall pass rate</h3>
      <p class="big">{_pct(report.overall_pass_rate)}</p>
    </div>
    <div class="card">
      <h3>Total cases</h3>
      <p class="big">{report.total_cases}</p>
    </div>
    <div class="card">
      <h3>Metrics / case</h3>
      <p class="big">{metrics_per_case}</p>
    </div>
    <div class="card">
      <h3>Weak probes</h3>
      <p class="big" style="color:{'#c0392b' if weak else '#1e8449'}">{weak}</p>
    </div>
  </div>"""


def _render_metric_coverage(report) -> str:
    """
    Full-suite coverage: the mean of every metric computed across all cases.

    This is what shows that the run exercised the broad metric catalogue (toxicity,
    hallucination, judge, fairness, …), not just the pass/fail gate metrics.
    """
    totals: dict = {}
    counts: dict = {}
    for r in report.results:
        for name, value in r.scores.items():
            if isinstance(value, (int, float)):
                totals[name] = totals.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1
    if not totals:
        return ""

    rows = []
    for name in sorted(totals):
        mean = totals[name] / counts[name]
        rows.append(
            f'<tr><td><code>{escape(name)}</code></td>'
            f'<td style="min-width:160px">{_bar(mean)}</td>'
            f'<td>{counts[name]}</td></tr>'
        )

    return f"""
  <section>
    <h2>Full metric coverage — {len(totals)} metrics</h2>
    <p class="sub">Mean score per metric across all {report.total_cases} cases. The agent runs every
    registry metric applicable to plain text output; metrics needing traces, documents or timing are skipped.</p>
    <details open>
      <summary>{len(totals)} metrics computed</summary>
      <table>
        <tr><th>Metric</th><th>Mean score</th><th>Cases</th></tr>
        {''.join(rows)}
      </table>
    </details>
  </section>"""


def _render_rounds(report) -> str:
    """
    The headline visualization: round-by-round adaptive drilling.

    Shows each round as a stage with the probes it ran and its pass rate, and calls
    out which probes were *dropped* between rounds (settled = passed) versus carried
    forward (drilled deeper).
    """
    if not report.rounds:
        return ""

    stages = []
    prev_probes = None
    for rnd in report.rounds:
        current = set(rnd.probes_run)

        # Which probes settled (present last round, gone this round)?
        dropped = sorted(prev_probes - current) if prev_probes is not None else []
        dropped_html = ""
        if dropped:
            chips = "".join(f'<span class="chip chip-settled">{escape(p)} ✓</span>' for p in dropped)
            dropped_html = f'<div class="dropped">Settled &amp; dropped after previous round: {chips}</div>'

        probe_chips = "".join(
            f'<span class="chip">{escape(p)}</span>' for p in rnd.probes_run
        )

        stages.append(f"""
    <div class="round">
      <div class="round-head">
        <span class="round-num">Round {rnd.round_number}</span>
        <span class="round-stat">{rnd.passed}/{rnd.cases_run} cases passed · {len(rnd.probes_run)} probe(s)</span>
      </div>
      {dropped_html}
      <div class="chips">{probe_chips}</div>
      {_bar(rnd.pass_rate)}
    </div>""")
        prev_probes = current

    note = (
        '<p class="sub">Each round, probes that pass (≥ 80%) are dropped and the agent '
        'mutates the failing inputs into harder variants — so the funnel narrows onto real weaknesses.</p>'
    )
    return f"""
  <section>
    <h2>Adaptive drilling — round by round</h2>
    {note}
    <div class="rounds">{''.join(stages)}</div>
  </section>"""


def _render_findings(report) -> str:
    """Per-probe findings table, sorted worst-first, with sample failing inputs."""
    findings = sorted(report.findings, key=lambda f: (f.pass_rate, -_sev_rank(f.severity)))
    if not findings:
        return ""

    rows = []
    for f in findings:
        status = (
            '<span class="tag tag-weak">WEAK</span>' if f.is_weak
            else '<span class="tag tag-ok">OK</span>'
        )
        weakest = (
            f"{escape(f.weakest_metric)} ({f.weakest_metric_score:.2f})"
            if f.weakest_metric else "—"
        )
        # Sample failing inputs — escaped because attack strings contain markup-like text.
        if f.failing_inputs:
            items = "".join(f"<li>{escape(s)}</li>" for s in f.failing_inputs)
            failing = f'<details><summary>{len(f.failing_inputs)} sample(s)</summary><ul>{items}</ul></details>'
        else:
            failing = "—"

        rows.append(f"""
      <tr>
        <td><code>{escape(f.probe)}</code></td>
        <td><span class="sev {_severity_class(f.severity)}">{escape(f.severity)}</span></td>
        <td>{escape(f.capability)}</td>
        <td>{f.cases_run}</td>
        <td style="min-width:140px">{_bar(f.pass_rate)}</td>
        <td>{weakest}</td>
        <td>{status}</td>
        <td>{failing}</td>
      </tr>""")

    return f"""
  <section>
    <h2>Per-probe findings</h2>
    <table>
      <tr>
        <th>Probe</th><th>Severity</th><th>Capability</th><th>Cases</th>
        <th>Pass rate</th><th>Weakest metric</th><th>Status</th><th>Failing inputs</th>
      </tr>
      {''.join(rows)}
    </table>
  </section>"""


def _sev_rank(severity: str) -> int:
    """Numeric severity rank so critical findings sort above lower ones at equal pass rate."""
    return {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(severity, 1)


# ============================================================================
# STYLES
# ============================================================================

_CSS = """<style>
  :root { --bg:#0f1419; --card:#1a212b; --ink:#e6edf3; --muted:#9bafc4; --accent:#4da3ff; --line:#2b3645; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  header { padding:30px 24px; background:linear-gradient(135deg,#16263b,#1a3a2e); border-bottom:1px solid var(--line); position:relative; }
  header h1 { margin:8px 0 4px; font-size:24px; }
  header .meta { margin:0; color:var(--muted); font-size:14px; }
  .verdict-badge { position:absolute; top:26px; right:26px; padding:8px 20px; border-radius:8px; font-weight:700; letter-spacing:1px; font-size:18px; }
  .verdict-pass { color:#1e8449; }
  .verdict-fail { color:#c0392b; }
  .verdict-badge.verdict-pass { background:#103a26; color:#48d597; }
  .verdict-badge.verdict-fail { background:#3a1414; color:#ff6b6b; }
  main { max-width:1080px; margin:0 auto; padding:24px; }
  section { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 24px; margin:20px 0; }
  h2 { margin:0 0 6px; font-size:19px; color:var(--accent); }
  .sub { color:var(--muted); font-size:13px; margin:0 0 16px; }
  .cards { display:flex; gap:16px; flex-wrap:wrap; margin:20px 0 0; }
  .card { flex:1; min-width:160px; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; text-align:center; }
  .card h3 { margin:0 0 8px; font-size:13px; color:var(--muted); font-weight:600; }
  .card .big { margin:0; font-size:30px; font-weight:700; color:var(--accent); }
  .summary-box { background:#0c1116; border:1px solid var(--line); border-radius:10px; padding:14px 16px; color:var(--muted); font-size:14px; }
  .rounds { display:flex; flex-direction:column; gap:14px; }
  .round { background:#0c1116; border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:10px; padding:14px 16px; }
  .round-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }
  .round-num { font-weight:700; font-size:15px; }
  .round-stat { color:var(--muted); font-size:13px; }
  .dropped { color:var(--muted); font-size:12px; margin:6px 0; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
  .chip { background:#16202c; border:1px solid var(--line); border-radius:20px; padding:3px 11px; font-size:12px; color:var(--ink); }
  .chip-settled { background:#103a26; color:#48d597; border-color:#1e5a3a; }
  .bar { position:relative; background:#16202c; border-radius:6px; height:20px; overflow:hidden; margin-top:6px; }
  .bar-fill { height:100%; border-radius:6px; transition:width .3s; }
  .bar-label { position:absolute; right:8px; top:0; line-height:20px; font-size:12px; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,.6); }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }
  th { background:#10202f; color:var(--accent); padding:9px 10px; text-align:left; }
  td { padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:middle; }
  code { background:#0c1116; border:1px solid var(--line); border-radius:5px; padding:1px 6px; font-size:12px; color:#a9e0c0; }
  .sev { padding:2px 9px; border-radius:12px; font-size:11px; font-weight:700; text-transform:uppercase; }
  .sev-critical { background:#3a1414; color:#ff6b6b; }
  .sev-high { background:#3a2814; color:#ffaa5b; }
  .sev-medium { background:#14283a; color:#5bb0ff; }
  .sev-low { background:#222; color:#bbb; }
  .tag { padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700; }
  .tag-weak { background:#3a1414; color:#ff6b6b; }
  .tag-ok { background:#103a26; color:#48d597; }
  details summary { cursor:pointer; color:var(--accent); font-size:12px; }
  details ul { margin:8px 0 0; padding-left:18px; color:var(--muted); }
  details li { margin:3px 0; font-size:12px; word-break:break-word; }
  footer { text-align:center; color:var(--muted); padding:26px; font-size:12px; }
</style>"""
