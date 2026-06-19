"""
evalgrid/report_html.py: Self-contained HTML report rendering for EvalRun.

No external assets, no JavaScript dependencies — the report opens in any browser
and prints cleanly on paper. Designed to be the first thing a stakeholder sees.
"""

from html import escape


def render_html_report(run, title: str = "EvalGrid Report") -> str:
    """Render an EvalRun as a standalone HTML document."""
    pass_rate_pct = f"{run.pass_rate * 100:.0f}%"
    passed_count  = sum(1 for r in run.results if r.passed)
    total         = len(run.results)
    status_colour = "#22c55e" if run.passed else "#ef4444"
    status_text   = "PASSED" if run.passed else "FAILED"

    metric_rows = ""
    for metric, mean in run.metric_averages().items():
        colour = "#22c55e" if mean >= run.threshold else "#ef4444"
        bar_width = int(min(max(mean, 0), 1) * 100)
        metric_rows += f"""
        <tr>
          <td class="metric-name">{escape(metric)}</td>
          <td class="metric-bar">
            <div class="bar-bg"><div class="bar-fill" style="width:{bar_width}%;background:{colour}"></div></div>
          </td>
          <td class="metric-value" style="color:{colour}">{mean:.3f}</td>
        </tr>"""

    case_rows = ""
    for r in run.results:
        status_badge = (
            '<span class="badge pass">PASS</span>' if r.passed
            else '<span class="badge fail">FAIL</span>'
        )
        scores_html = "".join(
            f'<span class="score-chip">{escape(m)}: <b>{v:.2f}</b></span>'
            for m, v in r.scores.items()
        )
        case_rows += f"""
        <tr>
          <td>{escape(r.test_id)}</td>
          <td>{status_badge}</td>
          <td class="case-input">{escape((r.input or '')[:200])}</td>
          <td class="case-output">{escape((r.output or '')[:200])}</td>
          <td class="case-scores">{scores_html}</td>
        </tr>"""

    cost_html = ""
    if run.cost:
        cost_html = f"""
        <div class="card">
          <div class="card-label">Estimated Cost</div>
          <div class="card-value">${run.cost.get('cost_usd', 0):.4f}</div>
          <div class="card-sub">{run.cost.get('calls', 0)} LLM calls · {run.cost.get('total_tokens', 0):,} tokens</div>
        </div>"""

    cache_html = ""
    if run.cache_stats and run.cache_stats.get("total"):
        cache_html = f"""
        <div class="card">
          <div class="card-label">Cache</div>
          <div class="card-value">{run.cache_stats.get('hit_rate', 0) * 100:.0f}%</div>
          <div class="card-sub">{run.cache_stats.get('hits', 0)} hits · {run.cache_stats.get('misses', 0)} misses</div>
        </div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #f8fafc; color: #0f172a; padding: 32px; max-width: 1200px; margin: 0 auto;
  }}
  header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }}
  h1 {{ font-size: 28px; font-weight: 700; }}
  .status {{ font-weight: 700; font-size: 16px; padding: 6px 16px; border-radius: 8px; color: white; }}
  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 24px;
  }}
  .card {{
    background: white; padding: 18px; border-radius: 12px; border: 1px solid #e2e8f0;
  }}
  .card-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card-value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
  .card-sub {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
  section {{
    background: white; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; margin-bottom: 20px;
  }}
  section h2 {{ font-size: 16px; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
  th {{ font-weight: 600; color: #64748b; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }}
  .metric-name {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; width: 40%; }}
  .metric-bar {{ width: 45%; }}
  .bar-bg {{ background: #f1f5f9; border-radius: 4px; height: 8px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .metric-value {{ font-weight: 700; text-align: right; width: 15%; }}
  .badge {{ font-weight: 700; font-size: 10px; padding: 3px 8px; border-radius: 4px; color: white; }}
  .badge.pass {{ background: #22c55e; }}
  .badge.fail {{ background: #ef4444; }}
  .case-input, .case-output {{ font-size: 12px; max-width: 280px; }}
  .case-output {{ color: #475569; }}
  .case-scores {{ font-size: 11px; max-width: 320px; }}
  .score-chip {{
    display: inline-block; background: #f1f5f9; border-radius: 6px;
    padding: 2px 6px; margin: 2px; font-family: ui-monospace, monospace;
  }}
  footer {{ text-align: center; font-size: 11px; color: #94a3b8; margin-top: 24px; }}
</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <div class="status" style="background:{status_colour}">{status_text}</div>
</header>

<div class="summary-grid">
  <div class="card">
    <div class="card-label">Pass Rate</div>
    <div class="card-value" style="color:{status_colour}">{pass_rate_pct}</div>
    <div class="card-sub">{passed_count} of {total} cases passed</div>
  </div>
  <div class="card">
    <div class="card-label">Threshold</div>
    <div class="card-value">{run.threshold}</div>
    <div class="card-sub">minimum acceptable score</div>
  </div>
  <div class="card">
    <div class="card-label">Metrics</div>
    <div class="card-value">{len(run.metrics_used)}</div>
    <div class="card-sub">{', '.join(run.metrics_used[:3])}{('…' if len(run.metrics_used) > 3 else '')}</div>
  </div>
  {cost_html}
  {cache_html}
</div>

<section>
  <h2>Metric Averages</h2>
  <table>
    <thead><tr><th>Metric</th><th>Distribution</th><th>Score</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>
</section>

<section>
  <h2>Per-Case Results ({total})</h2>
  <table>
    <thead><tr><th>ID</th><th>Status</th><th>Input</th><th>Output</th><th>Scores</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
</section>

<footer>Generated by EvalGrid · {escape(title)}</footer>
</body>
</html>"""
