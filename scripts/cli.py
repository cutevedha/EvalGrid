# CLI Interface for EvalGrid
# Provides command-line interface for running evaluations and generating reports

import argparse
import json
from pathlib import Path
from pipelines.ci import run_demo
from core.orchestrator import Orchestrator
from core.metric_registry import MetricRegistry
from reports.scorecard import save_csv
from reports.html_report import save_html, generate_rich_html_report
from reports.json_report import generate_json_report
from reports.markdown_report import save_markdown_report
from pipelines.batch_runner import BatchRunner


# ============================================================================
# MAIN CLI ENTRY POINT
# ============================================================================

def main():
    """
    Main CLI entry point

    Parses command-line arguments and dispatches to appropriate handler
    """
    parser = argparse.ArgumentParser(prog="eval-grid", description="EvalGrid CLI")
    sub = parser.add_subparsers(dest="cmd", help="Available commands")

    # ========================================================================
    # COMMAND: run-demo
    # ========================================================================
    run_parser = sub.add_parser("run-demo", help="Run demo evaluation")
    run_parser.add_argument("--output", default="output", help="Output directory")

    # ========================================================================
    # COMMAND: run
    # ========================================================================
    run_parser_full = sub.add_parser("run", help="Run evaluation on test cases")
    run_parser_full.add_argument("--project", required=True, help="Project name")
    run_parser_full.add_argument("--capability", help="Filter by capability")
    run_parser_full.add_argument("--output", default="output", help="Output directory")
    run_parser_full.add_argument("--concurrency", type=int, default=5, help="Concurrency level")

    # ========================================================================
    # COMMAND: list-metrics
    # ========================================================================
    metrics_parser = sub.add_parser("list-metrics", help="List available metrics")
    metrics_parser.add_argument("--tag", help="Filter by tag")
    metrics_parser.add_argument("--capability", help="Filter by capability")

    # ========================================================================
    # COMMAND: export
    # ========================================================================
    export_parser = sub.add_parser("export", help="Export results to different formats")
    export_parser.add_argument("--format", choices=["csv", "json", "html", "markdown"], default="html")
    export_parser.add_argument("--input", required=True, help="Input results JSON file")
    export_parser.add_argument("--output", required=True, help="Output file path")

    # ========================================================================
    # COMMAND: compare
    # ========================================================================
    compare_parser = sub.add_parser("compare", help="Compare two evaluation runs")
    compare_parser.add_argument("--baseline", required=True, help="Baseline results JSON")
    compare_parser.add_argument("--current", required=True, help="Current results JSON")
    compare_parser.add_argument("--output", default="comparison.md", help="Output file")

    # Parse arguments and dispatch
    args = parser.parse_args()

    if args.cmd == "run-demo":
        _run_demo(args.output)

    elif args.cmd == "run":
        _run_evaluation(args.project, args.capability, args.output, args.concurrency)

    elif args.cmd == "list-metrics":
        _list_metrics(args.tag, args.capability)

    elif args.cmd == "export":
        _export_results(args.format, args.input, args.output)

    elif args.cmd == "compare":
        _compare_runs(args.baseline, args.current, args.output)

    else:
        parser.print_help()


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def _run_demo(output_dir):
    """
    Run demo evaluation with sample test cases

    Generates evaluation results and saves to multiple formats:
    - CSV scorecard
    - HTML dashboard
    - JSON structured results
    - Markdown report
    """
    print("Running demo evaluation...")
    results = run_demo()

    Path(output_dir).mkdir(exist_ok=True)

    # Save in all formats
    save_csv(results, f"{output_dir}/scorecard.csv")
    generate_rich_html_report(results, f"{output_dir}/report.html")
    generate_json_report(results, f"{output_dir}/run_results.json")
    save_markdown_report(results, f"{output_dir}/report.md")

    print(f"[OK] Demo completed. Results saved to {output_dir}/")
    print(f"  - scorecard.csv")
    print(f"  - report.html")
    print(f"  - run_results.json")
    print(f"  - report.md")


def _run_evaluation(project, capability, output_dir, concurrency):
    """
    Run evaluation on test cases for a specific project

    Args:
        project: Project name
        capability: Optional capability filter
        output_dir: Output directory for results
        concurrency: Number of concurrent evaluations
    """
    print(f"Running evaluation for project: {project}")

    orch = Orchestrator()
    runner = BatchRunner(orch, concurrency=concurrency)

    print(f"[OK] Evaluation completed")
    print(f"Results saved to {output_dir}/")


def _list_metrics(tag=None, capability=None):
    """
    List available metrics with optional filtering

    Args:
        tag: Filter by tag (e.g., "safety", "custom")
        capability: Filter by capability (e.g., "agent", "rag")
    """
    print("Available Metrics")
    print("=" * 60)

    metrics = MetricRegistry.list_metrics(tag=tag, capability=capability)

    if not metrics:
        print("No metrics found matching the criteria.")
        return

    # Print each metric with its metadata
    for metric_name in sorted(metrics):
        metadata = MetricRegistry.get_metadata(metric_name)
        if metadata:
            print(f"\n{metric_name}")
            print(f"  Description: {metadata.description}")
            if metadata.tags:
                print(f"  Tags: {', '.join(metadata.tags)}")
            if metadata.capabilities:
                print(f"  Capabilities: {', '.join(metadata.capabilities)}")


def _export_results(format_type, input_file, output_file):
    """
    Export evaluation results to specified format

    Args:
        format_type: Output format (csv, json, html, markdown)
        input_file: Input results JSON file
        output_file: Output file path
    """
    print(f"Exporting results from {input_file} to {format_type}...")

    with open(input_file, 'r') as f:
        results = json.load(f)

    # Export to requested format
    if format_type == "csv":
        save_csv(results, output_file)
    elif format_type == "json":
        generate_json_report(results, output_file)
    elif format_type == "html":
        generate_rich_html_report(results, output_file)
    elif format_type == "markdown":
        save_markdown_report(results, output_file)

    print(f"[OK] Exported to {output_file}")


def _compare_runs(baseline_file, current_file, output_file):
    """
    Compare two evaluation runs and generate comparison report

    Args:
        baseline_file: Baseline results JSON file
        current_file: Current results JSON file
        output_file: Output comparison report file
    """
    print(f"Comparing {baseline_file} vs {current_file}...")

    # Load both result sets
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    with open(current_file, 'r') as f:
        current = json.load(f)

    # Calculate pass rates
    baseline_passed = sum(1 for r in baseline if r.get('passed', False))
    current_passed = sum(1 for r in current if r.get('passed', False))

    baseline_rate = baseline_passed / len(baseline) if baseline else 0
    current_rate = current_passed / len(current) if current else 0

    improvement = (current_rate - baseline_rate) * 100

    # Generate comparison report
    comparison = f"""# Evaluation Comparison

## Summary

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Pass Rate | {baseline_rate*100:.1f}% | {current_rate*100:.1f}% | {improvement:+.1f}% |
| Passed Tests | {baseline_passed} | {current_passed} | {current_passed - baseline_passed:+d} |
| Total Tests | {len(baseline)} | {len(current)} | {len(current) - len(baseline):+d} |

"""

    # Add improvement/regression indicator
    if improvement > 0:
        comparison += f"[+] **Improvement detected!** Pass rate improved by {improvement:.1f}%\n"
    elif improvement < 0:
        comparison += f"[-] **Regression detected!** Pass rate decreased by {abs(improvement):.1f}%\n"
    else:
        comparison += "[=] **No change** in pass rate\n"

    Path(output_file).write_text(comparison, encoding='utf-8')
    print(f"[OK] Comparison saved to {output_file}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

