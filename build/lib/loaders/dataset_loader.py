"""
loaders/dataset_loader.py: Load EvalGrid test cases from any common file format.

Supported formats
-----------------
| Extension          | Format  | Notes                                      |
|--------------------|---------|---------------------------------------------|
| .xlsx / .xls       | Excel   | Requires: pip install openpyxl              |
| .json              | JSON    | List or {"cases": [...]} wrapper            |
| .jsonl / .ndjson   | JSONL   | One JSON object per line                    |
| .csv               | CSV     | Standard comma-separated; pandas or stdlib  |
| .yaml / .yml       | YAML    | Requires: pip install pyyaml               |

Column name flexibility
-----------------------
Many common column aliases are accepted automatically. Examples:
  "question" / "prompt" / "query"         → mapped to  input
  "answer" / "reference" / "ground_truth" → mapped to  expected_output
  "document" / "passage" / "source"       → mapped to  context
  "system" / "system_message"             → mapped to  system_prompt
  "behavior" / "expected_action"          → mapped to  expected_behavior

Quick start
-----------
    from loaders import load_dataset

    # Auto-detect format from file extension
    cases = load_dataset("test_cases.xlsx")

    # Explicit format + sheet name (for Excel)
    cases = load_dataset("data.xlsx", sheet_name="Adversarial Tests")

    # JSON, JSONL, CSV, YAML all work the same way
    cases = load_dataset("my_tests.json")
    cases = load_dataset("my_tests.csv")
    cases = load_dataset("my_tests.yaml")

    # Then use cases directly in an Orchestrator or Governance pipeline
    from core.orchestrator import Orchestrator
    orch = Orchestrator()
    for case in cases:
        result = orch.run(case, model_output="...")
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schemas import TestCase


# ============================================================================
# COLUMN-NAME ALIASES
# Maps common column names → canonical TestCase field names.
# Add new aliases here without touching the rest of the loader.
# ============================================================================

_COLUMN_ALIASES: Dict[str, str] = {
    # input
    "question":     "input",
    "prompt":       "input",
    "query":        "input",
    "user_input":   "input",
    "user_message": "input",
    "message":      "input",
    "text":         "input",
    # expected_output
    "answer":           "expected_output",
    "reference":        "expected_output",
    "ground_truth":     "expected_output",
    "expected_answer":  "expected_output",
    "gold_answer":      "expected_output",
    "golden_answer":    "expected_output",
    "label":            "expected_output",
    "correct_answer":   "expected_output",
    # context
    "document":     "context",
    "passage":      "context",
    "source":       "context",
    "background":   "context",
    "knowledge":    "context",
    "retrieved":    "context",
    # system_prompt
    "system":           "system_prompt",
    "system_message":   "system_prompt",
    "instruction":      "system_prompt",
    "instructions":     "system_prompt",
    # expected_behavior
    "behavior":         "expected_behavior",
    "expected_action":  "expected_behavior",
    "outcome":          "expected_behavior",
    "expected_outcome": "expected_behavior",
    # capability
    "type":     "capability",
    "category": "capability",
    "task":     "capability",
    "task_type":"capability",
    # severity
    "priority":   "severity",
    "risk_level": "severity",
    "level":      "severity",
    # risk_tags
    "tags":      "risk_tags",
    "risk":      "risk_tags",
    "risk_tag":  "risk_tags",
    # project
    "project_name": "project",
    "suite":        "project",
    "test_suite":   "project",
}

_VALID_CAPABILITIES = frozenset({
    "generation", "extraction", "rag", "classification",
    "agent", "tool_use", "multi_agent", "embedded_ai",
})
_VALID_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
_VALID_EVAL_MODES = frozenset({"deterministic", "semantic", "judge", "hybrid"})

_KNOWN_FIELDS = frozenset({
    "id", "project", "capability", "input", "context",
    "expected_output", "expected_json", "risk_tags", "severity",
    "evaluation_mode", "thresholds", "expected_behavior", "system_prompt",
})

# ============================================================================
# PUBLIC API
# ============================================================================

def load_dataset(
    filepath: str,
    format: str = "auto",
    sheet_name: Optional[str] = None,
) -> List[TestCase]:
    """
    Load test cases from a file and return a list of TestCase objects.

    Args:
        filepath:   Path to the data file.
        format:     One of "auto", "excel", "json", "jsonl", "csv", "yaml".
                    "auto" detects the format from the file extension.
        sheet_name: For Excel files — sheet name or index (default: first sheet).

    Returns:
        List[TestCase] ready to pass to Orchestrator or GovernancePipeline.

    Raises:
        ValueError: if the format cannot be detected, a required column is missing,
                    or a row cannot be mapped to a valid TestCase.
        FileNotFoundError: if the file does not exist.
    """
    rows = load_dataset_raw(filepath, format=format, sheet_name=sheet_name)
    return _rows_to_test_cases(rows)


def load_dataset_raw(
    filepath: str,
    format: str = "auto",
    sheet_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Like load_dataset() but returns raw dicts rather than TestCase objects.

    Useful when you want to inspect or transform the data before evaluation,
    or when feeding rows directly to the GovernancePipeline runner/scorer.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    resolved_format = _detect_format(path.suffix.lower()) if format == "auto" else format

    loaders = {
        "excel": lambda: _load_excel(filepath, sheet_name=sheet_name),
        "json":  lambda: _load_json(filepath),
        "jsonl": lambda: _load_jsonl(filepath),
        "csv":   lambda: _load_csv(filepath),
        "yaml":  lambda: _load_yaml(filepath),
    }
    loader = loaders.get(resolved_format)
    if loader is None:
        raise ValueError(
            f"Unsupported format '{resolved_format}'. "
            "Choose from: excel, json, jsonl, csv, yaml (or set format='auto')."
        )
    return loader()


# ============================================================================
# FORMAT-SPECIFIC LOADERS
# ============================================================================

def _load_excel(filepath: str, sheet_name=None) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for Excel. Install with: pip install pandas openpyxl"
        )
    kwargs: Dict[str, Any] = {}
    if sheet_name is not None:
        kwargs["sheet_name"] = sheet_name
    df = pd.read_excel(filepath, **kwargs)
    # Replace NaN/NaT with None so rows serialise cleanly
    import numpy as np
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def _load_json(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Accept wrapper objects like {"cases": [...]} or {"test_cases": [...]}
    for key in ("cases", "test_cases", "data", "samples", "tests", "items", "examples"):
        if isinstance(data, dict) and key in data:
            return data[key]
    raise ValueError(
        f"JSON must be a list of test cases or a dict with a recognised wrapper key "
        f"(cases, test_cases, data, samples, tests, items, examples). "
        f"Got: {type(data).__name__} with keys {list(data.keys()) if isinstance(data, dict) else '—'}"
    )


def _load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {filepath}: {exc}") from exc
    return rows


def _load_csv(filepath: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(filepath, encoding="utf-8")
        df = df.where(pd.notna(df), None)
        return df.to_dict(orient="records")
    except ImportError:
        # Stdlib fallback — always available
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]


def _load_yaml(filepath: str) -> List[Dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    for key in ("cases", "test_cases", "data", "samples", "tests", "examples"):
        if isinstance(data, dict) and key in data:
            return data[key]
    raise ValueError(
        "YAML must be a list of test cases or a dict with a recognised wrapper key."
    )


# ============================================================================
# ROW → TestCase CONVERSION
# ============================================================================

def _rows_to_test_cases(rows: List[Dict[str, Any]]) -> List[TestCase]:
    test_cases: List[TestCase] = []
    for index, row in enumerate(rows):
        normalised = _normalise_row(row, index)
        try:
            test_cases.append(TestCase(**normalised))
        except Exception as exc:
            raise ValueError(
                f"Row {index + 1} could not be converted to a TestCase: {exc}\n"
                f"Normalised row: {normalised}"
            ) from exc
    return test_cases


def _normalise_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Apply alias mapping, fill required defaults, coerce types, strip unknown fields."""
    normalised: Dict[str, Any] = {}

    for raw_key, value in row.items():
        # Skip None / NaN values
        if value is None:
            continue
        try:
            if value != value:  # NaN check (float('nan') != float('nan'))
                continue
        except TypeError:
            pass

        canonical = _COLUMN_ALIASES.get(str(raw_key).lower().strip(), str(raw_key).lower().strip())
        normalised[canonical] = value

    # ── Required defaults ────────────────────────────────────────────────────
    if "id" not in normalised:
        normalised["id"] = f"case_{index + 1}"
    else:
        normalised["id"] = str(normalised["id"])

    if "project" not in normalised:
        normalised["project"] = "imported"

    # ── Capability coercion ──────────────────────────────────────────────────
    raw_cap = str(normalised.get("capability", "generation")).lower().strip()
    normalised["capability"] = raw_cap if raw_cap in _VALID_CAPABILITIES else "generation"

    # ── Severity coercion ────────────────────────────────────────────────────
    raw_sev = str(normalised.get("severity", "medium")).lower().strip()
    normalised["severity"] = raw_sev if raw_sev in _VALID_SEVERITIES else "medium"

    # ── Evaluation mode coercion ─────────────────────────────────────────────
    if "evaluation_mode" in normalised:
        raw_mode = str(normalised["evaluation_mode"]).lower().strip()
        normalised["evaluation_mode"] = raw_mode if raw_mode in _VALID_EVAL_MODES else "hybrid"

    # ── risk_tags: accept comma/semicolon-delimited strings ─────────────────
    risk_tags = normalised.get("risk_tags", [])
    if isinstance(risk_tags, str):
        normalised["risk_tags"] = [
            t.strip() for t in risk_tags.replace(";", ",").split(",") if t.strip()
        ]
    elif not isinstance(risk_tags, list):
        normalised["risk_tags"] = []

    # ── Validate required input field ────────────────────────────────────────
    if "input" not in normalised or not str(normalised.get("input", "")).strip():
        raise ValueError(
            "Row is missing a required 'input' value. "
            "Accepted aliases: question, prompt, query, user_input, user_message, text."
        )

    # ── Strip fields TestCase does not know about ────────────────────────────
    return {k: v for k, v in normalised.items() if k in _KNOWN_FIELDS}


def _detect_format(suffix: str) -> str:
    mapping = {
        ".xlsx": "excel", ".xls": "excel",
        ".json": "json",
        ".jsonl": "jsonl", ".ndjson": "jsonl",
        ".csv": "csv",
        ".yaml": "yaml", ".yml": "yaml",
    }
    if suffix not in mapping:
        raise ValueError(
            f"Cannot auto-detect format from extension '{suffix}'. "
            "Supported: .xlsx, .xls, .json, .jsonl, .csv, .yaml, .yml  — "
            "or pass format= explicitly."
        )
    return mapping[suffix]
