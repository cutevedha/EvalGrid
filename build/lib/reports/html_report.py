"""
reports/html_report.py: Export evaluation results as interactive HTML dashboards.

Provides two output styles:
- save_html()                : a minimal, dependency-free HTML table.  Useful for
                                quick inspection or embedding in other pages.
- generate_rich_html_report(): a full styled dashboard with colour-coded pass/fail
                                rows, metric score columns, and a summary header.
                                Self-contained (no external CSS/JS dependencies).

The HTML files can be opened directly in any web browser with no server needed.
"""

from html import escape
from pathlib import Path
from typing import List, Dict, Any


def _h(value: Any) -> str:
    """
    Escape a value for safe inclusion in HTML.

    Test IDs, scores and other fields can carry evaluation inputs or model output
    (including red-team attack strings). Escaping every interpolated value before it
    reaches the HTML prevents stored cross-site scripting when the report is opened
    in a browser.
    """
    return escape(str(value), quote=True)


# ============================================================================
# MINIMAL HTML REPORT
# ============================================================================

def save_html(results, path):
    """
    Generate a simple HTML table of evaluation results.

    Produces a minimal three-column table (Test ID, Passed, Scores).
    Use generate_rich_html_report() for a styled dashboard.

    Args:
        results: List of EvalResult.model_dump() dicts
        path: Output file path (e.g. "output/report.html")

    Returns:
        The path that was written
    """
    rows = []
    for r in results:
        rows.append(
            f"<tr>"
            f"<td>{_h(r.get('test_id'))}</td>"
            f"<td>{_h(r.get('passed'))}</td>"
            f"<td>{_h(r.get('scores'))}</td>"
            f"</tr>"
        )

    html = (
        f"<html><head><title>EvalGrid Report</title></head>"
        f"<body><h1>EvalGrid Report</h1>"
        f"<table border='1'>"
        f"<tr><th>Test ID</th><th>Passed</th><th>Scores</th></tr>"
        f"{''.join(rows)}"
        f"</table></body></html>"
    )
    Path(path).write_text(html, encoding="utf-8")
    return path



# ============================================================================
# RICH HTML DASHBOARD
# ============================================================================

def generate_rich_html_report(results: List[Dict[str, Any]], path: str, title: str = "EvalGrid Report") -> str:
    """
    Generate a styled HTML dashboard with summary cards, metrics table, and test detail.

    Sections:
    - Summary cards: total / passed / failed / pass-rate
    - Metrics summary table: mean, min, max, count per metric
    - Test results table: per-test pass/fail status and scores

    Args:
        results: List of EvalResult.model_dump() dicts
        path: Output file path (e.g. "output/report.html")
        title: Page heading shown in the browser tab and H1

    Returns:
        The path that was written
    """
    passed_count = sum(1 for r in results if r.get('passed', False))
    total_count  = len(results)
    pass_rate    = (passed_count / total_count * 100) if total_count > 0 else 0

    # ── Aggregate per-metric statistics across all results ────────────────────
    metric_stats: Dict[str, list] = {}
    for result in results:
        for metric_name, score in result.get('scores', {}).items():
            if metric_name not in metric_stats:
                metric_stats[metric_name] = []
            metric_stats[metric_name].append(score)

    metric_summaries = {}
    for metric_name, scores in metric_stats.items():
        metric_summaries[metric_name] = {
            "mean":  sum(scores) / len(scores),
            "min":   min(scores),
            "max":   max(scores),
            "count": len(scores),
        }

    # ── Summary cards HTML ────────────────────────────────────────────────────
    summary_cards = f"""
    <div class="summary-cards">
        <div class="card">
            <h3>Total Tests</h3>
            <p class="metric">{total_count}</p>
        </div>
        <div class="card">
            <h3>Passed</h3>
            <p class="metric" style="color: green;">{passed_count}</p>
        </div>
        <div class="card">
            <h3>Failed</h3>
            <p class="metric" style="color: red;">{total_count - passed_count}</p>
        </div>
        <div class="card">
            <h3>Pass Rate</h3>
            <p class="metric">{pass_rate:.1f}%</p>
        </div>
    </div>
    """

    # ── Per-metric summary table HTML ─────────────────────────────────────────
    metric_rows = []
    for metric_name, stats in metric_summaries.items():
        metric_rows.append(f"""
        <tr>
            <td>{_h(metric_name)}</td>
            <td>{stats['mean']:.3f}</td>
            <td>{stats['min']:.3f}</td>
            <td>{stats['max']:.3f}</td>
            <td>{stats['count']}</td>
        </tr>
        """)

    metrics_table = f"""
    <h2>Metrics Summary</h2>
    <table class="metrics-table">
        <tr>
            <th>Metric</th>
            <th>Mean</th>
            <th>Min</th>
            <th>Max</th>
            <th>Count</th>
        </tr>
        {''.join(metric_rows)}
    </table>
    """

    # ── Per-test results table HTML ───────────────────────────────────────────
    test_rows = []
    for r in results:
        status = (
            '<span style="color: green;">&#10003; PASS</span>'
            if r.get('passed')
            else '<span style="color: red;">&#10007; FAIL</span>'
        )
        scores_html = '<br>'.join([f"{_h(k)}: {v:.3f}" for k, v in r.get('scores', {}).items()])
        test_rows.append(f"""
        <tr>
            <td>{_h(r.get('test_id'))}</td>
            <td>{status}</td>
            <td>{scores_html}</td>
        </tr>
        """)

    tests_table = f"""
    <h2>Test Results</h2>
    <table class="tests-table">
        <tr>
            <th>Test ID</th>
            <th>Status</th>
            <th>Scores</th>
        </tr>
        {''.join(test_rows)}
    </table>
    """

    # ── Inline CSS styles ─────────────────────────────────────────────────────
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1   { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        h2   { color: #555; margin-top: 30px; }
        .summary-cards { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
        .card {
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 150px; text-align: center;
        }
        .card h3 { margin: 0 0 10px 0; color: #666; }
        .metric  { font-size: 28px; font-weight: bold; color: #007bff; }
        table {
            width: 100%; border-collapse: collapse; background: white;
            margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th { background-color: #007bff; color: white; padding: 12px; text-align: left; }
        td { padding: 12px; border-bottom: 1px solid #ddd; }
        tr:hover { background-color: #f9f9f9; }
    </style>
    """

    # ── Assemble the full HTML document ───────────────────────────────────────
    safe_title = _h(title)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{safe_title}</title>
        {css}
    </head>
    <body>
        <h1>{safe_title}</h1>
        {summary_cards}
        {metrics_table}
        {tests_table}
    </body>
    </html>
    """

    Path(path).write_text(html, encoding="utf-8")
    return path

