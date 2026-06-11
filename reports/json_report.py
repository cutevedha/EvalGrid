"""
reports/json_report.py: Export evaluation results to structured JSON.

The JSON report is the most machine-readable output format.  It includes:
  - Metadata: timestamp, total cases, overall pass rate.
  - Per-metric summary: min, max, mean, and count across all results.
  - Full per-case results: scores, pass/fail status, and evaluator notes.

Use this format when you want to:
  - Feed results into a data pipeline or database.
  - Compare two runs programmatically with `eval-grid compare`.
  - Archive a full audit trail of an evaluation run.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def save_json_report(results: List[Dict[str, Any]], path: str) -> str:
    """
    Save evaluation results to a minimal JSON file with basic metadata.

    Args:
        results: List of EvalResult.model_dump() dicts
        path: Output file path

    Returns:
        The path that was written
    """
    report = {
        "metadata": {
            "total_tests":  len(results),
            "passed_tests": sum(1 for r in results if r.get('passed', False)),
            "failed_tests": sum(1 for r in results if not r.get('passed', False)),
        },
        "results": results,
    }

    Path(path).write_text(json.dumps(report, indent=2, cls=DateTimeEncoder))
    return path


def generate_json_report(results: List[Dict[str, Any]], path: str, include_metadata: bool = True) -> str:
    """
    Save evaluation results to a rich JSON file with per-metric statistics.

    In addition to the raw results, this function computes descriptive
    statistics (mean, min, max, std_dev) for every metric across all tests.

    Args:
        results: List of EvalResult.model_dump() dicts
        path: Output file path
        include_metadata: Reserved for future use (always included)

    Returns:
        The path that was written
    """
    # Collect all scores per metric across all results
    metric_stats: Dict[str, List[float]] = {}
    for result in results:
        for metric_name, score in result.get('scores', {}).items():
            if metric_name not in metric_stats:
                metric_stats[metric_name] = []
            metric_stats[metric_name].append(score)

    # Compute descriptive statistics for each metric
    metric_summaries = {}
    for metric_name, scores in metric_stats.items():
        metric_summaries[metric_name] = {
            "mean":    sum(scores) / len(scores) if scores else 0.0,
            "min":     min(scores) if scores else 0.0,
            "max":     max(scores) if scores else 0.0,
            "std_dev": _calculate_stddev(scores),
            "count":   len(scores),
        }

    report = {
        "metadata": {
            "total_tests":  len(results),
            "passed_tests": sum(1 for r in results if r.get('passed', False)),
            "failed_tests": sum(1 for r in results if not r.get('passed', False)),
            "pass_rate":    sum(1 for r in results if r.get('passed', False)) / len(results) if results else 0.0,
        },
        "metric_summaries": metric_summaries,
        "results": results,
    }

    Path(path).write_text(json.dumps(report, indent=2, cls=DateTimeEncoder))
    return path


def _calculate_stddev(values: List[float]) -> float:
    """
    Compute population standard deviation.

    Args:
        values: List of numeric values

    Returns:
        Standard deviation (0.0 if fewer than 2 values)
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

