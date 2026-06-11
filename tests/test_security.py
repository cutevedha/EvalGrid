"""
Security regression tests.

These lock in fixes for two report-layer vulnerabilities so they cannot silently
return: stored XSS in HTML reports (CWE-79) and CSV formula injection (CWE-1236).
Evaluation inputs and model outputs are untrusted, so anything that flows into a
report must be neutralised.
"""

import tempfile
from pathlib import Path

from reports.html_report import save_html, generate_rich_html_report
from reports.scorecard import save_csv


_XSS = "<script>alert(1)</script>"
_IMG = '"><img src=x onerror=alert(1)>'


def _results(test_id, score_key="metric"):
    return [{"test_id": test_id, "passed": False, "scores": {score_key: 0.0}, "notes": []}]


def test_minimal_html_escapes_payload(tmp_path):
    out = tmp_path / "min.html"
    save_html(_results(_XSS), str(out))
    html = out.read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_rich_html_escapes_test_id_metric_and_title(tmp_path):
    out = tmp_path / "rich.html"
    generate_rich_html_report(_results(_XSS, score_key=_IMG), str(out), title=_XSS)
    html = out.read_text(encoding="utf-8")
    # No executable markup survives: tag openings from untrusted fields are escaped.
    assert "<script>alert" not in html
    assert "<img src=x" not in html       # the injected <img> tag must be neutralised
    assert "&lt;img src=x" in html        # ...and present only as inert escaped text


def test_csv_neutralises_formula_injection(tmp_path):
    out = tmp_path / "scores.csv"
    rows = [{"test_id": "=1+1", "passed": True, "scores": {}, "notes": ["@SUM(A1)"]},
            {"test_id": "+cmd", "passed": True, "scores": {}, "notes": ["-2+3"]}]
    save_csv(rows, str(out))
    csv = out.read_text(encoding="utf-8")
    # Each formula trigger is prefixed with a single quote so spreadsheets treat it as text.
    for trigger in ("'=1+1", "'@SUM(A1)", "'+cmd", "'-2+3"):
        assert trigger in csv


def test_csv_leaves_safe_values_untouched(tmp_path):
    out = tmp_path / "safe.csv"
    save_csv([{"test_id": "normal_id", "passed": True, "scores": {}, "notes": ["ok"]}], str(out))
    csv = out.read_text(encoding="utf-8")
    assert "normal_id" in csv and "'normal_id" not in csv
