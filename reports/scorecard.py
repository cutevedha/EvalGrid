# Scorecard Report - Exports evaluation results to CSV format
# Produces a flat spreadsheet suitable for analysis in Excel or pandas

import pandas as pd


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

    Args:
        results: List of EvalResult.model_dump() dicts
        path: File path to write (e.g. "output/scorecard.csv")

    Returns:
        The path that was written
    """
    df = to_dataframe(results)
    df.to_csv(path, index=False)  # No row index column in the output
    return path
