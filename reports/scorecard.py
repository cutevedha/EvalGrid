"""
reports/scorecard.py: Export evaluation results to a flat CSV spreadsheet.

A CSV scorecard is the easiest format to open in Excel, Google Sheets, or load
into pandas for further analysis (e.g. filtering by severity, plotting score
distributions, or comparing model versions).

Each row represents one test case.  Columns include:
  - test_id, passed, evaluator_version
  - One column per metric score
  - Notes from the evaluator

Requires pandas: pip install pandas
"""

import pandas as pd

# Leading characters a spreadsheet (Excel, Google Sheets, LibreOffice) treats as the
# start of a formula. A test_id or note beginning with one of these could execute when
# the CSV is opened, so such cells are neutralised before writing.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _neutralise_formula(value):
    """Prefix a leading formula trigger with a single quote so spreadsheets treat it as text."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def to_dataframe(results):
    """
    Convert a list of evaluation result dicts into a pandas DataFrame.

    Each row in the DataFrame represents one test case result.
    Nested dicts (e.g. 'scores') are kept as-is in a single column.

    Args:
        results: List of EvalResult.model_dump() dicts

    Returns:
        pandas DataFrame with one row per test case
    """
    return pd.DataFrame(results)


def save_csv(results, path):
    """
    Serialise evaluation results to a CSV file.

    String cells that begin with a spreadsheet formula trigger are neutralised first to
    prevent CSV formula injection when the file is opened in Excel or Google Sheets.

    Args:
        results: List of EvalResult.model_dump() dicts
        path: File path to write (e.g. "output/scorecard.csv")

    Returns:
        The path that was written
    """
    df = to_dataframe(results)
    # Sanitise every string cell against formula injection (CWE-1236).
    df = df.map(_neutralise_formula) if hasattr(df, "map") else df.applymap(_neutralise_formula)
    df.to_csv(path, index=False)  # No row index column in the output
    return path
