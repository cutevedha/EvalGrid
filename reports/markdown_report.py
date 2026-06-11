"""
reports/markdown_report.py: Export evaluation results as GitHub-Flavoured Markdown.

Markdown reports are ideal for:
  - Posting as a comment on a pull request to show evaluation results.
  - Including in project wikis or documentation pages.
  - Quick human review in any text editor or GitHub/GitLab UI.

The report contains a summary table with pass/fail status and key metric scores
for each test case, plus an overall pass rate header.
"""

from pathlib import Path
from typing import List, Dict, Any


def save_markdown_report(results: List[Dict[str, Any]], path: str, title: str = "EvalGrid Report") -> str:
    """
    Generate a full Markdown evaluation report with summary, metrics table, and per-test detail.

    Args:
        results: List of EvalResult.model_dump() dicts
        path: Output file path (e.g. "output/report.md")
        title: Report heading

    Returns:
        The path that was written
    """
    passed_count = sum(1 for r in results if r.get('passed', False))
    total_count  = len(results)
    pass_rate    = (passed_count / total_count * 100) if total_count > 0 else 0

    # ── Aggregate per-metric statistics ──────────────────────────────────────
    metric_stats: Dict[str, List[float]] = {}
    for result in results:
        for metric_name, score in result.get('scores', {}).items():
            if metric_name not in metric_stats:
                metric_stats[metric_name] = []
            metric_stats[metric_name].append(score)

    metric_summaries = {}
    for metric_name, scores in metric_stats.items():
        metric_summaries[metric_name] = {
            "mean":  sum(scores) / len(scores) if scores else 0.0,
            "min":   min(scores) if scores else 0.0,
            "max":   max(scores) if scores else 0.0,
            "count": len(scores),
        }

    # ── Build Markdown ────────────────────────────────────────────────────────
    markdown = f"""# {title}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {total_count} |
| Passed | {passed_count} |
| Failed | {total_count - passed_count} |
| Pass Rate | {pass_rate:.1f}% |

## Metrics Overview

| Metric | Mean | Min | Max | Count |
|--------|------|-----|-----|-------|
"""

    # One row per metric
    for metric_name, stats in metric_summaries.items():
        markdown += f"| {metric_name} | {stats['mean']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['count']} |\n"

    markdown += "\n## Test Results\n\n"

    # Separate passed and failed tests for clarity
    passed_tests = [r for r in results if r.get('passed', False)]
    failed_tests = [r for r in results if not r.get('passed', False)]

    if passed_tests:
        markdown += "### ✅ Passed Tests\n\n"
        for r in passed_tests:
            markdown += f"- **{r.get('test_id')}**\n"

    if failed_tests:
        markdown += "\n### ❌ Failed Tests\n\n"
        for r in failed_tests:
            markdown += f"- **{r.get('test_id')}**\n"
            for note in r.get('notes', []):
                markdown += f"  - {note}\n"

    # Full detail section
    markdown += "\n## Detailed Results\n\n"
    for r in results:
        status = "✅ PASS" if r.get('passed', False) else "❌ FAIL"
        markdown += f"### {r.get('test_id')} - {status}\n\n"
        markdown += "**Scores:**\n\n"
        for metric_name, score in r.get('scores', {}).items():
            markdown += f"- {metric_name}: {score:.3f}\n"
        markdown += "\n"

    Path(path).write_text(markdown, encoding='utf-8')
    return path


def generate_pr_comment(results: List[Dict[str, Any]]) -> str:
    """
    Generate a concise Markdown summary suitable for posting as a pull-request comment.

    Lists up to 5 failed tests and indicates improvement/regression/no-change.

    Args:
        results: List of EvalResult.model_dump() dicts

    Returns:
        Markdown string (not written to a file)
    """
    passed_count = sum(1 for r in results if r.get('passed', False))
    total_count  = len(results)
    pass_rate    = (passed_count / total_count * 100) if total_count > 0 else 0

    comment = f"""## 🤖 EvalGrid Report

**Pass Rate:** {pass_rate:.1f}% ({passed_count}/{total_count})

| Status | Count |
|--------|-------|
| ✅ Passed | {passed_count} |
| ❌ Failed | {total_count - passed_count} |

"""

    failed_tests = [r for r in results if not r.get('passed', False)]
    if failed_tests:
        comment += "### Failed Tests\n\n"
        for r in failed_tests[:5]:  # Show at most 5 failures to keep the comment compact
            comment += f"- {r.get('test_id')}\n"
        if len(failed_tests) > 5:
            comment += f"- ... and {len(failed_tests) - 5} more\n"

    return comment
