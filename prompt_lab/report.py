"""
prompt_lab/report.py: Generate an HTML report and a plain-text summary.

The HTML report is designed to be opened by anyone — no coding required.
It shows:
  - A top-level verdict badge (PASS / PARTIAL / FAIL)
  - Side-by-side scores per LLM with colour-coded cells
  - Full LLM responses in expandable sections
  - The auto-generated fixed prompt (when available)
  - A copy-to-clipboard button on the fixed prompt
"""

from __future__ import annotations

import datetime
import html as html_lib
from pathlib import Path
from typing import List

from prompt_lab.evaluator import PromptReport, EvalResult, DIMENSIONS


VERDICT_COLOUR = {"PASS": "#22c55e", "PARTIAL": "#f59e0b", "FAIL": "#ef4444", "SKIPPED": "#94a3b8", "UNKNOWN": "#94a3b8"}
SCORE_BG = {
    "high":   "#bbf7d0",  # >= 7
    "mid":    "#fef08a",  # >= 5
    "low":    "#fecaca",  # < 5
    "na":     "#f1f5f9",
}


def _score_bg(val: float) -> str:
    if val >= 7:
        return SCORE_BG["high"]
    if val >= 5:
        return SCORE_BG["mid"]
    return SCORE_BG["low"]


def _e(text: str) -> str:
    """HTML-escape helper."""
    return html_lib.escape(str(text))


def _score_table(results: List[EvalResult]) -> str:
    active = [r for r in results if not r.skipped]
    if not active:
        return "<p><em>No LLMs were configured. Add API keys to your .env file.</em></p>"

    headers = ["Dimension"] + [r.llm for r in active]
    rows_html = ""
    for dim in DIMENSIONS:
        cells = f"<td><strong>{dim.capitalize()}</strong></td>"
        for r in active:
            val = getattr(r.scores, dim)
            bg = _score_bg(val)
            cells += f'<td style="background:{bg};text-align:center;font-weight:bold">{val}/10</td>'
        rows_html += f"<tr>{cells}</tr>"

    # Average row
    avg_cells = "<td><strong>Average</strong></td>"
    for r in active:
        avg = r.scores.average
        bg = _score_bg(avg)
        avg_cells += f'<td style="background:{bg};text-align:center;font-weight:bold">{avg}/10</td>'
    rows_html += f"<tr>{avg_cells}</tr>"

    # Verdict row
    verdict_cells = "<td><strong>Verdict</strong></td>"
    for r in active:
        colour = VERDICT_COLOUR.get(r.scores.verdict, "#94a3b8")
        verdict_cells += (
            f'<td style="background:{colour};color:white;text-align:center;font-weight:bold">'
            f'{r.scores.verdict}</td>'
        )
    rows_html += f"<tr>{verdict_cells}</tr>"

    # Latency row
    lat_cells = "<td>Latency</td>"
    for r in active:
        lat_cells += f'<td style="text-align:center">{r.latency_ms} ms</td>'
    rows_html += f"<tr>{lat_cells}</tr>"

    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    return f"""
    <table>
      <thead><tr>{th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def _observations(results: List[EvalResult]) -> str:
    parts = []
    for r in results:
        if r.skipped:
            parts.append(
                f'<div class="obs skipped"><strong>{_e(r.llm)}</strong> — '
                f'<em>Skipped: {_e(r.skip_reason)}</em></div>'
            )
        else:
            colour = VERDICT_COLOUR.get(r.scores.verdict, "#94a3b8")
            parts.append(
                f'<div class="obs" style="border-left:4px solid {colour}">'
                f'<strong>{_e(r.llm)}</strong> ({r.scores.verdict}): '
                f'{_e(r.scores.observations)}</div>'
            )
    return "\n".join(parts)


def _response_sections(results: List[EvalResult]) -> str:
    parts = []
    for r in results:
        if r.skipped:
            continue
        colour = VERDICT_COLOUR.get(r.scores.verdict, "#94a3b8")
        parts.append(f"""
        <details>
          <summary style="cursor:pointer;font-weight:bold;color:{colour}">
            {_e(r.llm)} response ({r.latency_ms} ms)
          </summary>
          <pre class="response">{_e(r.response)}</pre>
        </details>""")
    return "\n".join(parts)


def generate_html(report: PromptReport, output_path: str) -> str:
    """Write an HTML report to output_path. Returns the path."""
    verdict_colour = VERDICT_COLOUR.get(report.overall_verdict, "#94a3b8")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    fixed_section = ""
    if report.fixed_prompt:
        fixed_id = "fixed-prompt-text"
        fixed_section = f"""
        <section>
          <h2>&#128295; Suggested Fixed Prompt</h2>
          <p>The following improved prompt was generated automatically based on the observations above.
             Click <strong>Copy</strong> to use it.</p>
          <pre id="{fixed_id}" class="response fixed">{_e(report.fixed_prompt)}</pre>
          <button onclick="copyFixed()">&#128203; Copy to clipboard</button>
          <span id="copy-msg" style="margin-left:1em;color:green;display:none">Copied!</span>
        </section>"""

    skipped_warning = ""
    skipped = [r for r in report.results if r.skipped]
    if skipped:
        names = ", ".join(r.llm for r in skipped)
        skipped_warning = (
            f'<div class="banner warn">&#9888; {names} was not tested — '
            f'API key missing or SDK not installed. See observations below.</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prompt Lab Report — {_e(report.prompt_title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0; padding: 2rem; background: #f8fafc; color: #1e293b; }}
    h1   {{ margin-bottom: 0.25rem; }}
    h2   {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-top: 2rem; }}
    .meta  {{ color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }}
    .badge {{ display: inline-block; padding: 0.4rem 1.2rem; border-radius: 9999px;
              color: white; font-weight: bold; font-size: 1.1rem;
              background: {verdict_colour}; margin-bottom: 1rem; }}
    .banner {{ padding: 0.8rem 1.2rem; border-radius: 8px; margin-bottom: 1.5rem; }}
    .banner.warn {{ background: #fef3c7; border: 1px solid #f59e0b; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 0.6rem 1rem; }}
    th {{ background: #1e293b; color: white; }}
    .obs {{ background: white; padding: 0.75rem 1rem; margin: 0.5rem 0;
            border-radius: 6px; border-left: 4px solid #94a3b8; }}
    .obs.skipped {{ border-left: 4px solid #94a3b8; color: #64748b; }}
    pre.response {{ background: #1e293b; color: #e2e8f0; padding: 1.2rem;
                   border-radius: 8px; white-space: pre-wrap; word-break: break-word;
                   font-size: 0.88rem; max-height: 500px; overflow-y: auto; }}
    pre.fixed {{ background: #0f172a; border: 2px solid #22c55e; }}
    details {{ margin: 1rem 0; }}
    summary {{ padding: 0.5rem; border-radius: 4px; }}
    summary:hover {{ background: #f1f5f9; }}
    button {{ background: #2563eb; color: white; border: none; padding: 0.5rem 1.2rem;
              border-radius: 6px; cursor: pointer; font-size: 0.95rem; }}
    button:hover {{ background: #1d4ed8; }}
    .prompt-box {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px;
                   padding: 1.2rem; white-space: pre-wrap; font-family: monospace;
                   font-size: 0.9rem; max-height: 300px; overflow-y: auto; }}
    section {{ background: white; border-radius: 10px; padding: 1.5rem;
               margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
  <h1>&#128203; Prompt Lab Report</h1>
  <div class="meta">
    <strong>Prompt:</strong> {_e(report.prompt_title)} &nbsp;|&nbsp;
    <strong>ID:</strong> {_e(report.prompt_id)} &nbsp;|&nbsp;
    <strong>Generated:</strong> {now}
  </div>

  <div class="badge">Overall: {_e(report.overall_verdict)}</div>
  {skipped_warning}

  <section>
    <h2>&#128220; Original Prompt</h2>
    <div class="prompt-box">{_e(report.prompt_text)}</div>
  </section>

  <section>
    <h2>&#127775; Scores</h2>
    {_score_table(report.results)}
  </section>

  <section>
    <h2>&#128269; Observations</h2>
    {_observations(report.results)}
  </section>

  <section>
    <h2>&#128172; Full Responses</h2>
    {_response_sections(report.results)}
  </section>

  {fixed_section}

  <script>
    function copyFixed() {{
      var text = document.getElementById("fixed-prompt-text").innerText;
      navigator.clipboard.writeText(text).then(function() {{
        var msg = document.getElementById("copy-msg");
        msg.style.display = "inline";
        setTimeout(function() {{ msg.style.display = "none"; }}, 2000);
      }});
    }}
  </script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output_path


def print_summary(report: PromptReport) -> None:
    """Print a concise terminal summary."""
    colour_map = {"PASS": "\033[92m", "PARTIAL": "\033[93m", "FAIL": "\033[91m"}
    reset = "\033[0m"
    c = colour_map.get(report.overall_verdict, "")

    print(f"\n{'='*60}")
    print(f"  Prompt : {report.prompt_title}")
    print(f"  ID     : {report.prompt_id}")
    print(f"  Verdict: {c}{report.overall_verdict}{reset}")
    print(f"{'='*60}")

    for r in report.results:
        if r.skipped:
            print(f"  {r.llm:10s}  SKIPPED — {r.skip_reason}")
        else:
            vc = colour_map.get(r.scores.verdict, "")
            print(
                f"  {r.llm:10s}  {vc}{r.scores.verdict:7s}{reset}  "
                f"avg {r.scores.average}/10   ({r.latency_ms} ms)"
            )
            print(f"             {r.scores.observations}")

    if report.fixed_prompt:
        print(f"\n  {colour_map['PARTIAL']}A fixed prompt was generated and included in the HTML report.{reset}")
    print()
