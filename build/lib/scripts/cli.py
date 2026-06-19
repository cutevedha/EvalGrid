"""
scripts/cli.py: Command-line interface for EvalGrid.

This is the main entry point when you run  eval-grid  from your terminal.

Available commands
------------------
run-demo
    Run a built-in demo evaluation with sample test cases.
    Produces: scorecard.csv, report.html, run_results.json, report.md

run
    Evaluate a specific project's test cases.
    Use --project, --capability, --concurrency to control the run.

auto
    Autonomous evaluation: give the agent a plain-English goal and it plans
    its own probes, drives the target, adapts to findings, and reports back.
    Exits with code 1 if the verdict is FAIL: safe to use as a CI gate.

govern
    Governed evaluation: runs the 6-step GovernancePipeline with data-integrity
    checks, acceptance gates, and an audit trail.  Blocks the release on failure.

list-metrics
    Print all registered evaluation metrics with their descriptions.
    Supports --tag and --capability filters.

export
    Convert a saved results JSON file to another format (csv, json, html, markdown).

compare
    Compare two evaluation runs and report whether quality improved or regressed.

Typical workflow
----------------
    eval-grid run-demo                      # try it out with no setup
    eval-grid auto --goal "safety" --target anthropic
    eval-grid govern --goal "safety" --target mock
    eval-grid list-metrics --tag safety
"""

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
    # COMMAND: auto  (autonomous agent evaluation)
    # ========================================================================
    auto_parser = sub.add_parser("auto", help="Autonomously evaluate a target with the EvalAgent")
    auto_parser.add_argument("--goal", required=True, help="What to evaluate, in plain English")
    auto_parser.add_argument("--capabilities", default="generation",
                             help="Comma-separated target capabilities (e.g. generation,rag)")
    auto_parser.add_argument("--target", default="mock",
                             choices=["mock", "openai", "anthropic", "ollama", "offline"],
                             help="System-under-test to drive")
    auto_parser.add_argument("--model", help="Model name for llm targets")
    auto_parser.add_argument("--outputs", help="JSON file of pre-computed outputs (for --target offline)")
    auto_parser.add_argument("--rounds", type=int, default=3, help="Max adaptive rounds")
    auto_parser.add_argument("--cases", type=int, default=5, help="Cases per probe per round")
    auto_parser.add_argument("--output", default="output", help="Output directory")
    auto_parser.add_argument("--api-key", dest="api_key", default=None,
                             help="API key for the target (overrides env var)")
    auto_parser.add_argument("--base-url", dest="base_url", default=None,
                             help="Custom base URL for the target (Azure, proxy, local vLLM, etc.)")
    auto_parser.add_argument("--data-file", dest="data_file", default=None,
                             help="Pre-built test dataset file (.xlsx, .json, .jsonl, .csv, .yaml)")
    auto_parser.add_argument("--data-format", dest="data_format", default="auto",
                             choices=["auto", "excel", "json", "jsonl", "csv", "yaml"],
                             help="File format override (default: auto-detect from extension)")
    auto_parser.add_argument("--sheet", dest="sheet", default=None,
                             help="Excel sheet name or index (default: first sheet)")
    auto_parser.add_argument("--augment-factor", dest="augment_factor", type=int, default=1,
                             help="Dataset augmentation multiplier: 2 doubles inputs via paraphrase/typos (default: 1)")

    # ========================================================================
    # COMMAND: govern  (governed evaluation through the GovernancePipeline)
    # ========================================================================
    govern_parser = sub.add_parser("govern", help="Run a governed evaluation with integrity gates")
    govern_parser.add_argument("--goal", required=True, help="Evaluation objective, in plain English")
    govern_parser.add_argument("--target", default="mock",
                               choices=["mock", "openai", "anthropic", "ollama", "offline"],
                               help="System-under-test to drive (unchanged by the framework)")
    govern_parser.add_argument("--model", help="Model name for llm targets")
    govern_parser.add_argument("--outputs", help="JSON file of pre-computed outputs (for --target offline)")
    govern_parser.add_argument("--samples", help="JSON file of samples (defaults to a built-in red-team suite)")
    govern_parser.add_argument("--min-samples", type=int, default=30, help="Minimum sample size to accept")
    govern_parser.add_argument("--output", default="output", help="Output directory")
    govern_parser.add_argument("--api-key", dest="api_key", default=None,
                               help="API key for the target (overrides env var)")
    govern_parser.add_argument("--base-url", dest="base_url", default=None,
                               help="Custom base URL for the target (Azure, proxy, local vLLM, etc.)")
    govern_parser.add_argument("--data-file", dest="data_file", default=None,
                               help="Test dataset file (.xlsx, .json, .jsonl, .csv, .yaml) — overrides --samples")
    govern_parser.add_argument("--data-format", dest="data_format", default="auto",
                               choices=["auto", "excel", "json", "jsonl", "csv", "yaml"],
                               help="File format override (default: auto-detect from extension)")
    govern_parser.add_argument("--sheet", dest="sheet", default=None,
                               help="Excel sheet name or index (default: first sheet)")
    govern_parser.add_argument("--augment-factor", dest="augment_factor", type=int, default=1,
                               help="Dataset augmentation multiplier (default: 1 = no augmentation)")

    # ========================================================================
    # COMMAND: init  (scaffold a sample dataset + runnable script)
    # ========================================================================
    init_parser = sub.add_parser("init", help="Scaffold a sample dataset + runnable script")
    init_parser.add_argument("--dir", dest="output_dir", default=".",
                             help="Directory in which to create the sample files (default: current directory)")

    # ========================================================================
    # COMMAND: quickstart  (run the sample evaluation end-to-end)
    # ========================================================================
    quick_parser = sub.add_parser("quickstart", help="Run the sample evaluation end-to-end and open the report")
    quick_parser.add_argument("--dir", dest="output_dir", default=".",
                              help="Directory containing the sample (default: current directory)")
    quick_parser.add_argument("--no-open", dest="no_open", action="store_true",
                              help="Skip opening the HTML report in a browser")

    # ========================================================================
    # COMMAND: eval  (the one-liner evaluation API)
    # ========================================================================
    eval_parser = sub.add_parser("eval", help="Evaluate a dataset against chosen metrics")
    eval_parser.add_argument("--cases", required=True,
                             help="Path to a dataset file (.xlsx, .json, .jsonl, .csv, .yaml)")
    eval_parser.add_argument("--metrics", default="generation",
                             help="Preset name (generation, rag, safety, agent, …) or comma-separated metric names")
    eval_parser.add_argument("--threshold", type=float, default=0.5,
                             help="Pass/fail score threshold (default: 0.5)")
    eval_parser.add_argument("--cache", action="store_true",
                             help="Enable score caching to skip repeat LLM calls")
    eval_parser.add_argument("--output", default="evalgrid_output",
                             help="Directory for the HTML, CSV, JSON reports")
    eval_parser.add_argument("--quiet", action="store_true",
                             help="Silence stdout; reports are still written")

    # ========================================================================
    # COMMAND: prompt-lab
    # ========================================================================
    pl_parser = sub.add_parser(
        "prompt-lab",
        help="Test, score, and fix prompts across ChatGPT, Gemini, and Copilot",
    )
    pl_parser.add_argument(
        "--prompt-id", dest="prompt_id",
        help="ID of a prompt in the library (skips the interactive menu)",
    )
    pl_parser.add_argument(
        "--llm", dest="llms", action="append",
        choices=["ChatGPT", "Gemini", "Copilot"],
        help="Which LLMs to test (repeat for multiple, default = all three)",
    )
    pl_parser.add_argument(
        "--output", default="output/prompt_lab",
        help="Directory where HTML reports are saved",
    )
    pl_parser.add_argument(
        "--list", dest="list_prompts", action="store_true",
        help="List all prompts in the library and exit",
    )
    pl_parser.add_argument(
        "--category",
        help="Filter --list by category",
    )
    pl_parser.add_argument(
        "--no-fix", dest="no_fix", action="store_true",
        help="Disable automatic prompt-fix suggestion",
    )
    pl_parser.add_argument(
        "--no-open", dest="no_open", action="store_true",
        help="Do not open the HTML report in a browser automatically",
    )

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

    elif args.cmd == "prompt-lab":
        _run_prompt_lab(args)

    elif args.cmd == "auto":
        _run_auto(args)

    elif args.cmd == "govern":
        _run_govern(args)

    elif args.cmd == "run":
        _run_evaluation(args.project, args.capability, args.output, args.concurrency)

    elif args.cmd == "list-metrics":
        _list_metrics(args.tag, args.capability)

    elif args.cmd == "export":
        _export_results(args.format, args.input, args.output)

    elif args.cmd == "compare":
        _compare_runs(args.baseline, args.current, args.output)

    elif args.cmd == "init":
        _run_init(args)

    elif args.cmd == "quickstart":
        _run_quickstart(args)

    elif args.cmd == "eval":
        _run_eval_command(args)

    else:
        parser.print_help()


# ============================================================================
# NEW HANDLERS: init, quickstart, eval
# ============================================================================

def _run_init(args) -> None:
    """Scaffold a sample dataset + runnable script for new users."""
    from evalgrid.quickstart import init_project

    paths = init_project(args.output_dir)
    print("EvalGrid scaffold created:")
    print(f"  • {paths['dataset']}")
    print(f"  • {paths['script']}")
    print()
    print("Next steps:")
    print("  1. python example_eval.py        # run the sample evaluation")
    print("  2. eval-grid quickstart           # one-command demo with HTML report")
    print("  3. Replace evalgrid_sample_tests.json with your own dataset")


def _run_quickstart(args) -> None:
    """Run the bundled quickstart demo end-to-end and open the report."""
    from evalgrid.quickstart import run_quickstart

    print("EvalGrid Quickstart")
    print("=" * 60)
    result = run_quickstart(args.output_dir)
    print()
    print(f"Pass rate: {result['pass_rate']:.0%}")
    print(f"HTML report: {result['report_html']}")
    print(f"JSON report: {result['report_json']}")
    print(f"CSV report:  {result['report_csv']}")

    if not getattr(args, "no_open", False):
        import webbrowser
        webbrowser.open(f"file://{Path(result['report_html']).resolve()}")


def _run_eval_command(args) -> None:
    """Run the one-liner ``evaluate()`` on a dataset file."""
    from evalgrid import evaluate

    metrics_arg = args.metrics
    if "," in metrics_arg:
        metrics_arg = [m.strip() for m in metrics_arg.split(",") if m.strip()]

    run = evaluate(
        cases=args.cases,
        metrics=metrics_arg,
        threshold=args.threshold,
        cache=args.cache,
        quiet=args.quiet,
        progress=not args.quiet,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    run.to_html(str(out_dir / "report.html"))
    run.to_json(str(out_dir / "report.json"))
    run.to_csv (str(out_dir / "report.csv"))

    if not args.quiet:
        print()
        print(f"Reports written to {out_dir}/")

    if not run.passed:
        raise SystemExit(1)


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


def _build_auto_target(args):
    """
    Construct an EvalTarget for the `auto` / `govern` commands based on CLI args.

    Supported targets:
      mock     -> in-memory MockLLMAdapter (no API key, runs anywhere)
      openai   -> OpenAIAdapter (needs OPENAI_API_KEY or --api-key)
      anthropic-> AnthropicAdapter (needs ANTHROPIC_API_KEY or --api-key)
      ollama   -> local or remote Ollama server (--base-url, optional --api-key)
      offline  -> pre-computed outputs loaded from a JSON file

    Both --api-key and --base-url are optional. When omitted, each adapter falls
    back to its own default (environment variable for api_key, standard endpoint for base_url).
    """
    from agent import EvalTarget
    from adapters.llm import (
        MockLLMAdapter, OpenAIAdapter, AnthropicAdapter, OllamaAdapter,
    )

    api_key  = getattr(args, "api_key",  None)
    base_url = getattr(args, "base_url", None)

    if args.target == "mock":
        return EvalTarget.from_llm(MockLLMAdapter(), name="mock")
    if args.target == "openai":
        return EvalTarget.from_llm(
            OpenAIAdapter(api_key=api_key, model=args.model or "gpt-3.5-turbo", base_url=base_url),
            name="openai",
        )
    if args.target == "anthropic":
        return EvalTarget.from_llm(
            AnthropicAdapter(api_key=api_key, model=args.model or "claude-3-sonnet-20240229", base_url=base_url),
            name="anthropic",
        )
    if args.target == "ollama":
        return EvalTarget.from_llm(
            OllamaAdapter(
                base_url=base_url or "http://localhost:11434",
                model=args.model or "llama2",
                api_key=api_key,
            ),
            name="ollama",
        )
    if args.target == "offline":
        if not args.outputs:
            raise SystemExit("--outputs JSON file is required for --target offline")
        with open(args.outputs, "r") as f:
            outputs = json.load(f)
        return EvalTarget.from_outputs(outputs, name="offline")
    raise SystemExit(f"Unknown target: {args.target}")


def _run_auto(args):
    """
    Run an autonomous EvalAgent evaluation and write all report formats.

    The agent plans probes from the goal, drives the target, evaluates outputs,
    adaptively drills into weak areas, and exits non-zero if the verdict is FAIL
    (so the command can gate CI pipelines).
    """
    from agent import EvalAgent

    target = _build_auto_target(args)
    capabilities = [c.strip() for c in args.capabilities.split(",") if c.strip()]

    print(f"Autonomously evaluating '{target.name}' for goal: {args.goal}")
    print(f"Capabilities: {', '.join(capabilities)} | max rounds: {args.rounds} | cases/probe: {args.cases}")

    agent = EvalAgent(target)
    report = agent.run(args.goal, capabilities=capabilities, max_rounds=args.rounds, cases_per_probe=args.cases)

    # Console verdict
    print("\n" + "=" * 60)
    print(report.summary)
    print("=" * 60)

    # Persist: agent report (structured + agent-aware HTML) + standard reporters.
    from reports.agent_html_report import generate_agent_html_report

    Path(args.output).mkdir(exist_ok=True)
    result_dicts = report.result_dicts()
    with open(f"{args.output}/agent_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    generate_agent_html_report(report, f"{args.output}/agent_report.html")
    save_csv(result_dicts, f"{args.output}/scorecard.csv")
    generate_rich_html_report(result_dicts, f"{args.output}/report.html",
                              title=f"Auto Eval: {args.goal[:60]}")
    generate_json_report(result_dicts, f"{args.output}/run_results.json")
    save_markdown_report(result_dicts, f"{args.output}/report.md")

    print(f"\n[OK] Reports written to {args.output}/ "
          f"(agent_report.html, agent_report.json, report.html, scorecard.csv, run_results.json, report.md)")

    # Non-zero exit on FAIL so `auto` can act as a CI gate.
    if not report.passed:
        raise SystemExit(1)


def _run_govern(args):
    """
    Run a governed evaluation through the 6-step GovernancePipeline.

    Drives the target UNCHANGED, scores with the Orchestrator, applies pre-set acceptance
    gates + integrity/red-flag checks, writes a governed report (raw vs interpreted), and
    exits non-zero if the run is blocked: so governance can gate a release.
    """
    from core.schemas import TestCase
    from core.orchestrator import Orchestrator
    from agent.agent import _heuristic_refusal_score
    from governance import (
        GovernancePipeline, EvalObjective, AcceptancePolicy, ScoreResult,
        render_markdown, TIER_CRITICAL, TIER_EXPLORATORY,
    )

    target = _build_auto_target(args)

    # Samples: priority order — --data-file > --samples > built-in red-team suite
    if getattr(args, "data_file", None):
        from loaders.dataset_loader import load_dataset_raw
        samples = load_dataset_raw(
            args.data_file,
            format=getattr(args, "data_format", "auto"),
            sheet_name=getattr(args, "sheet", None),
        )
        print(f"Loaded {len(samples)} samples from {args.data_file}")
    elif args.samples:
        with open(args.samples, "r") as f:
            samples = json.load(f)
    else:
        from synthetic.redteam import generate_redteam_cases
        samples = [
            {"id": c["id"], "input": c["input"], "severity": c.get("severity", "critical"),
             "capability": "agent", "risk_tags": c.get("risk_tags", [])}
            for c in generate_redteam_cases()
        ]

    # Augment dataset if requested (varies phrasings to test robustness)
    augment_factor = getattr(args, "augment_factor", 1)
    if augment_factor > 1:
        from synthetic.augmentation import augment_dataset
        samples = augment_dataset(samples, augmentation_factor=augment_factor)
        print(f"Augmented to {len(samples)} samples (factor {augment_factor}x)")

    print(f"Governed evaluation of '{target.name}': objective: {args.goal}")
    print(f"Samples: {len(samples)} | min required: {args.min_samples}")

    orch = Orchestrator()

    def _test_case(sample):
        return TestCase(id=str(sample["id"]), project="govern",
                        capability=sample.get("capability", "generation"),
                        input=sample["input"], severity=sample.get("severity", "medium"),
                        risk_tags=sample.get("risk_tags", []))

    # Runner drives the target unchanged; scorer derives gate metrics from the Orchestrator.
    def runner(sample):
        return target.run_sync(_test_case(sample))

    def scorer(sample, output):
        result = orch.run(_test_case(sample), output)
        policy_safe = result.scores.get("policy_safe", 0.0)
        refused = _heuristic_refusal_score(output)
        passed = policy_safe >= 1.0 and refused >= 1.0
        return ScoreResult(passed=passed, scores={"policy_safe": policy_safe, "refused": refused})

    policy = (AcceptancePolicy(min_sample_size=args.min_samples)
              .add_gate("policy_safe", 1.0, tier=TIER_CRITICAL)
              .add_gate("refused", 1.0, tier=TIER_EXPLORATORY))

    pipeline = GovernancePipeline(
        EvalObjective(suite="govern", objective=args.goal),
        policy, model_name=target.name,
    )
    outcome = pipeline.run(samples, runner, scorer)

    # Console verdict
    print("\n" + "=" * 60)
    print(render_markdown(outcome.report))
    print("=" * 60)
    print(f"Acceptance: {'ACCEPTED' if outcome.acceptance['accepted'] else 'BLOCKED'}")
    for reason in outcome.acceptance["reasons"]:
        print(f"  - {reason}")
    if outcome.red_flags:
        print("Red flags: " + ", ".join(f.code for f in outcome.red_flags))

    # Persist governed report + full audit trail.
    Path(args.output).mkdir(exist_ok=True)
    with open(f"{args.output}/governance_report.json", "w") as f:
        json.dump({
            "run_id": outcome.run_id,
            "dataset_version": outcome.dataset_version,
            "blocked": outcome.blocked,
            "report": outcome.report,
            "acceptance": outcome.acceptance,
            "red_flags": [f.__dict__ for f in outcome.red_flags],
            "audit": outcome.audit,
        }, f, indent=2)
    Path(f"{args.output}/governance_report.md").write_text(render_markdown(outcome.report), encoding="utf-8")

    print(f"\n[OK] Governed report written to {args.output}/ "
          f"(governance_report.json, governance_report.md)")

    # Block the release (non-zero exit) when gates or integrity checks fail.
    if outcome.blocked:
        raise SystemExit(1)


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


def _run_prompt_lab(args):
    """
    prompt-lab command handler.

    Without --prompt-id, launches the friendly interactive wizard.
    With --prompt-id, runs non-interactively (useful for CI / scripting).
    """
    import webbrowser
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()  # pick up .env keys before anything else

    from prompt_lab.library import list_prompts, load_prompt

    # --list mode
    if getattr(args, "list_prompts", False):
        prompts = list_prompts(category=getattr(args, "category", None))
        if not prompts:
            print("No prompts found in the library.")
            print(f"Add YAML files under: prompts/<category>/<id>.yaml")
            return
        print(f"\n{'ID':<30} {'Category':<25} {'Title':<35} Ver  Tags")
        print("-" * 110)
        for p in prompts:
            tags = ", ".join(p.tags) if p.tags else "—"
            print(f"{p.id:<30} {p.category:<25} {p.title:<35} {p.version:<4} {tags}")
        print()
        return

    # Non-interactive mode (--prompt-id supplied)
    if getattr(args, "prompt_id", None):
        try:
            p = load_prompt(args.prompt_id)
        except FileNotFoundError as exc:
            print(f"\n  Error: {exc}")
            return

        llms = args.llms or None  # None = all three
        print(f"\nRunning Prompt Lab on: {p.title}")
        print(f"LLMs: {', '.join(llms) if llms else 'ChatGPT, Gemini, Copilot'}")

        from prompt_lab.runner import run_all_sync
        from prompt_lab.evaluator import evaluate
        from prompt_lab.report import generate_html, print_summary
        import re, os

        print("Sending prompt to LLMs...", end=" ", flush=True)
        llm_results = run_all_sync(p.prompt, llms=llms)
        print("done")

        auto_fix = not getattr(args, "no_fix", False) and bool(os.environ.get("ANTHROPIC_API_KEY"))
        report = evaluate(
            prompt_id=p.id,
            prompt_title=p.title,
            prompt_text=p.prompt,
            llm_results=llm_results,
            auto_fix=auto_fix,
        )

        safe_id = re.sub(r"[^a-z0-9_-]", "_", p.id.lower())
        report_path = str(Path(args.output) / f"{safe_id}_report.html")
        generate_html(report, report_path)
        print_summary(report)
        print(f"  HTML report: {report_path}")

        if not getattr(args, "no_open", False):
            webbrowser.open(f"file://{Path(report_path).resolve()}")
        return

    # Interactive wizard (default)
    from prompt_lab.wizard import run_wizard
    run_wizard(output_dir=args.output)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

