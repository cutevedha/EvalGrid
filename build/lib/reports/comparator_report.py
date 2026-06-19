"""
reports/comparator_report.py: Export evaluation results as an LLM Comparator JSON file.

The LLM Comparator format lets you visualise and compare two model runs side-by-side
in an interactive dashboard (compatible with Google's LLM Comparator tool and similar).

Usage
-----
    from reports.comparator_report import generate_comparator_json

    # results_a and results_b are lists of eval result dicts (from Orchestrator or pipeline)
    generate_comparator_json(
        results_a=results_model_gpt4,
        results_b=results_model_claude,
        output_path="output/comparator.json",
        model_a_name="GPT-4",
        model_b_name="Claude Sonnet",
    )

Output format
-------------
The JSON has two top-level keys:
  models   — list of model labels
  examples — one entry per test case with both outputs and all metric scores

This format is compatible with Google's LLM Comparator:
  https://github.com/PAIR-code/llm-comparator
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_comparator_json(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    output_path: str,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> None:
    """
    Write a comparator-compatible JSON file from two parallel result lists.

    Args:
        results_a:    Eval results for the first model (list of dicts with "input",
                      "output", "scores", "passed", "test_id" keys).
        results_b:    Eval results for the second model (same format, same ordering).
        output_path:  File path for the JSON output.
        model_a_name: Display label for the first model.
        model_b_name: Display label for the second model.
    """
    # Index results_b by test_id for O(1) lookup when pairing
    index_b: Dict[str, Dict[str, Any]] = {r.get("test_id", str(i)): r for i, r in enumerate(results_b)}

    examples: List[Dict[str, Any]] = []
    for result_a in results_a:
        test_id = result_a.get("test_id", "")
        result_b = index_b.get(test_id, {})

        scores_a = result_a.get("scores", {})
        scores_b = result_b.get("scores", {})

        # Determine pairwise winner from scores if available
        avg_a = _safe_mean(list(scores_a.values()))
        avg_b = _safe_mean(list(scores_b.values()))
        if abs(avg_a - avg_b) < 0.05:
            winner = "tie"
        elif avg_a > avg_b:
            winner = "a"
        else:
            winner = "b"

        example: Dict[str, Any] = {
            "id":           test_id,
            "prompt":       result_a.get("input", ""),
            "tags":         result_a.get("risk_tags", []),
            # Model outputs
            "response_a": {
                "text":    result_a.get("output", ""),
                "passed":  result_a.get("passed", False),
                "scores":  scores_a,
            },
            "response_b": {
                "text":    result_b.get("output", ""),
                "passed":  result_b.get("passed", False),
                "scores":  scores_b,
            },
            # Aggregate pairwise preference (auto-computed from metric scores)
            "auto_winner": winner,
            # Per-metric comparison (which model scored higher on each metric)
            "metric_comparison": _compare_metrics(scores_a, scores_b),
        }
        examples.append(example)

    payload: Dict[str, Any] = {
        "models": [
            {"label": model_a_name, "description": f"Evaluation results for {model_a_name}"},
            {"label": model_b_name, "description": f"Evaluation results for {model_b_name}"},
        ],
        "examples": examples,
        "summary": {
            "model_a": model_a_name,
            "model_b": model_b_name,
            "total_examples": len(examples),
            "a_wins": sum(1 for e in examples if e["auto_winner"] == "a"),
            "b_wins": sum(1 for e in examples if e["auto_winner"] == "b"),
            "ties":   sum(1 for e in examples if e["auto_winner"] == "tie"),
            "avg_score_a": round(_safe_mean([
                _safe_mean(list(e["response_a"]["scores"].values())) for e in examples
            ]), 4),
            "avg_score_b": round(_safe_mean([
                _safe_mean(list(e["response_b"]["scores"].values())) for e in examples
            ]), 4),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _compare_metrics(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
) -> Dict[str, str]:
    """Return per-metric winner: "a", "b", or "tie"."""
    all_metrics = set(scores_a) | set(scores_b)
    comparison: Dict[str, str] = {}
    for metric in sorted(all_metrics):
        va = scores_a.get(metric, 0.0)
        vb = scores_b.get(metric, 0.0)
        if abs(va - vb) < 0.01:
            comparison[metric] = "tie"
        elif va > vb:
            comparison[metric] = "a"
        else:
            comparison[metric] = "b"
    return comparison


def _safe_mean(values: List[float]) -> float:
    valid = [v for v in values if isinstance(v, (int, float))]
    return sum(valid) / len(valid) if valid else 0.0
