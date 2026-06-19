"""
evalgrid/evaluate.py: The flagship one-liner evaluation API.

    from evalgrid import evaluate

    results = evaluate(cases=[...], metrics=[...])
    results.summary()          # text summary
    results.passed             # True if every case passed
    results.to_html("out.html")
    results.to_csv("out.csv")
    results.to_json("out.json")
    results.failed_cases()     # only the failures
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from core.metric_registry import MetricRegistry
from core.schemas import TestCase
from evalgrid.cache import ScoreCache
from evalgrid.cost import CostTracker
from evalgrid.presets import MetricSet, resolve_preset
from evalgrid.progress import progress_iter


# Type aliases — keep the public API approachable
CaseLike   = Union[TestCase, Dict[str, Any]]
MetricLike = Union[str, Callable, "object"]


# ============================================================================
# RESULT OBJECT
# ============================================================================

@dataclass
class CaseResult:
    """Per-test-case scores and pass/fail status."""
    test_id: str
    input: str
    output: str
    expected_output: Optional[str]
    scores: Dict[str, float]
    passed: bool
    threshold: float
    failed_metrics: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id":         self.test_id,
            "input":           self.input,
            "output":          self.output,
            "expected_output": self.expected_output,
            "scores":          self.scores,
            "passed":          self.passed,
            "threshold":       self.threshold,
            "failed_metrics":  self.failed_metrics,
            "notes":           self.notes,
        }


@dataclass
class EvalRun:
    """
    The friendly result object returned by ``evaluate()``.

    Iterate to access per-case results; check ``.passed`` for overall status;
    use ``.summary()`` or one of the export methods for reporting.
    """
    results: List[CaseResult]
    metrics_used: List[str]
    threshold: float
    cost: Optional[Dict[str, Any]] = None
    cache_stats: Optional[Dict[str, Any]] = None
    judge_stats: Optional[Dict[str, Any]] = None
    judge_model: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    concurrency: Optional[int] = None
    throughput: Optional[float] = None  # cases per second

    # ── Iteration / aggregation ────────────────────────────────────────────
    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx) -> CaseResult:
        return self.results[idx]

    @property
    def passed(self) -> bool:
        """True when every case passed."""
        return all(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed (0.0 – 1.0)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def failed_cases(self) -> List[CaseResult]:
        """Only the cases that failed — useful for debugging."""
        return [r for r in self.results if not r.passed]

    def metric_averages(self) -> Dict[str, float]:
        """Mean of each metric across every case."""
        by_metric: Dict[str, List[float]] = {}
        for result in self.results:
            for metric, value in result.scores.items():
                by_metric.setdefault(metric, []).append(value)
        return {metric: round(statistics.mean(vals), 4) for metric, vals in by_metric.items()}

    # ── Summary text ───────────────────────────────────────────────────────
    def summary(self) -> str:
        """Plain-text report suitable for stdout or logs."""
        lines: List[str] = []
        passed_count = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        symbol = "✓" if self.passed else "✗"
        lines.append(f"{symbol}  EvalGrid Run — {passed_count}/{total} passed ({self.pass_rate:.0%})")
        lines.append(f"   metrics: {', '.join(self.metrics_used)}")
        lines.append(f"   threshold: {self.threshold}")
        if self.elapsed_seconds is not None:
            tp = f" ({self.throughput:.1f} cases/s)" if self.throughput else ""
            conc = f" · concurrency={self.concurrency}" if self.concurrency else ""
            lines.append(f"   elapsed: {self.elapsed_seconds:.2f}s{tp}{conc}")
        if self.judge_model:
            lines.append(f"   judge: {self.judge_model}")
        if self.judge_stats and self.judge_stats.get("real_calls"):
            lines.append(
                f"   LLM calls: {self.judge_stats['real_calls']} "
                f"(cache hits: {self.judge_stats.get('cache_hits', 0)}, "
                f"errors: {self.judge_stats.get('errors', 0)})"
            )
        if self.cost:
            lines.append(f"   est. cost: ${self.cost.get('cost_usd', 0):.4f} ({self.cost.get('calls', 0)} tracked calls)")
        if self.cache_stats and self.cache_stats.get("total"):
            lines.append(f"   score cache: {self.cache_stats.get('hit_rate', 0):.0%} hit-rate ({self.cache_stats.get('hits', 0)} hits)")

        lines.append("")
        lines.append("Per-metric averages:")
        for metric, mean in self.metric_averages().items():
            indicator = "✓" if mean >= self.threshold else "✗"
            lines.append(f"   {indicator} {metric:35s}  {mean:.3f}")

        if not self.passed:
            lines.append("")
            lines.append(f"Failed cases ({len(self.failed_cases())}):")
            for result in self.failed_cases()[:5]:
                preview = (result.input or "")[:60].replace("\n", " ")
                lines.append(f"   • {result.test_id}: \"{preview}...\" — failed on {', '.join(result.failed_metrics)}")
            if len(self.failed_cases()) > 5:
                lines.append(f"   … and {len(self.failed_cases()) - 5} more")
        return "\n".join(lines)

    # ── Export ─────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed":        self.passed,
            "pass_rate":     self.pass_rate,
            "threshold":     self.threshold,
            "metrics_used":  self.metrics_used,
            "metric_averages": self.metric_averages(),
            "results":       [r.to_dict() for r in self.results],
            "cost":          self.cost,
            "cache_stats":   self.cache_stats,
        }

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_csv(self, path: str) -> None:
        import csv
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            return
        # Build header from union of all scores
        all_metrics = sorted({m for r in self.results for m in r.scores.keys()})
        fieldnames = ["test_id", "input", "output", "passed"] + all_metrics
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                row = {
                    "test_id": r.test_id,
                    "input":   r.input[:200],
                    "output":  r.output[:200],
                    "passed":  r.passed,
                }
                row.update({m: r.scores.get(m, "") for m in all_metrics})
                writer.writerow(row)

    def to_html(self, path: str, title: str = "EvalGrid Report") -> None:
        """Export an HTML report — minimal, no-JS, self-contained."""
        from evalgrid.report_html import render_html_report
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_html_report(self, title=title))


# ============================================================================
# THE FLAGSHIP evaluate() FUNCTION
# ============================================================================

def evaluate(
    cases: Union[Sequence[CaseLike], str],
    metrics: Union[str, List[MetricLike]],
    *,
    threshold: float = 0.5,
    cache: Union[bool, ScoreCache] = False,
    cost_tracker: Optional[CostTracker] = None,
    judge: Union[str, Any, bool, None] = None,
    concurrency: int = 10,
    batch_judging: bool = True,
    progress: bool = True,
    quiet: bool = False,
    fail_fast: bool = False,
) -> EvalRun:
    """
    Run an evaluation. The one-line entry point.

    Args:
        cases:        List of TestCase objects, list of plain dicts, or a path to
                      an Excel/JSON/JSONL/CSV/YAML file (auto-detected).
        metrics:      A preset name (e.g. "rag"), a MetricSet constant, or a list
                      of metric names / callables / BaseMetric instances.
        threshold:    Score below which a metric is considered a failure (default 0.5).
        cache:        Pass True to use ".evalgrid_cache", a ScoreCache instance to
                      use a custom cache, or False to disable caching (default False).
        cost_tracker: Optional CostTracker to capture LLM usage.
        judge:        LLM judge configuration:
                        - None (default): auto-detect from OPENAI_API_KEY / ANTHROPIC_API_KEY
                          if any LLM-judge metric is requested.
                        - "gpt-4o-mini" (string): use this model as the judge.
                        - False: force heuristic mode — never call any LLM.
                        - A JudgeClient or adapter instance: use it directly.
        concurrency:  Maximum number of cases evaluated in parallel (default 10).
                      Set to 1 for fully sequential / deterministic execution.
                      Higher values (50-100) give massive speedups on large datasets,
                      bounded by the LLM provider's rate limit.
        batch_judging: When True (default), multiple LLM-judge metrics for the same
                      case are scored in ONE LLM call instead of N calls — typically
                      a 5x cost reduction. Set False to disable for debugging.
        progress:     Show a progress bar (default True). Set False in tests / CI.
        quiet:        Silence stdout entirely.
        fail_fast:    Raise on first failure instead of accumulating all results.

    Returns:
        EvalRun — iterable, exportable, summarisable.

    Examples:
        # Minimal
        evaluate(cases=[TestCase(input="?", expected_output="!")], metrics=["correctness"])

        # Preset
        evaluate(cases=my_cases, metrics="rag")

        # From file
        evaluate(cases="tests.xlsx", metrics=MetricSet.SAFETY)

        # With caching + cost tracking
        evaluate(cases, metrics="generation", cache=True, cost_tracker=CostTracker("gpt-4o-mini"))
    """
    return _run_async_safe(
        _evaluate_async(
            cases=cases, metrics=metrics, threshold=threshold,
            cache=cache, cost_tracker=cost_tracker, judge=judge,
            concurrency=concurrency, batch_judging=batch_judging,
            progress=progress, quiet=quiet, fail_fast=fail_fast,
        )
    )


async def a_evaluate(
    cases: Union[Sequence[CaseLike], str],
    metrics: Union[str, List[MetricLike]],
    *,
    threshold: float = 0.5,
    cache: Union[bool, ScoreCache] = False,
    cost_tracker: Optional[CostTracker] = None,
    judge: Union[str, Any, bool, None] = None,
    concurrency: int = 10,
    batch_judging: bool = True,
    progress: bool = True,
    quiet: bool = False,
    fail_fast: bool = False,
) -> EvalRun:
    """
    Async variant of ``evaluate()`` — use directly in async code (FastAPI, Jupyter, etc.).

    Same signature and behavior as ``evaluate()``. Prefer this when you're already
    inside an event loop; ``evaluate()`` will work too but pays a thread-handoff cost.
    """
    return await _evaluate_async(
        cases=cases, metrics=metrics, threshold=threshold,
        cache=cache, cost_tracker=cost_tracker, judge=judge,
        concurrency=concurrency, batch_judging=batch_judging,
        progress=progress, quiet=quiet, fail_fast=fail_fast,
    )


# ============================================================================
# ASYNC CORE — concurrent case evaluation
# ============================================================================

async def _evaluate_async(
    *,
    cases, metrics, threshold, cache, cost_tracker, judge,
    concurrency, batch_judging, progress, quiet, fail_fast,
) -> EvalRun:
    """
    Asynchronous core of the evaluation engine.

    Each case is run on a worker via ``asyncio.to_thread`` so the existing
    synchronous metric code keeps working unchanged. Concurrency is bounded by a
    semaphore so we never exceed the LLM provider's rate limit.
    """
    test_cases   = _coerce_cases(cases)
    metric_names = _resolve_metric_list(metrics)
    score_cache  = _resolve_cache(cache)
    show_progress = progress and not quiet

    previous_judge_state = _configure_judge_for_run(judge, metric_names, cost_tracker)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    start_time = time.time()

    # Identify which LLM-judge metrics can be batched into one call per case.
    batchable_metrics: List[str] = []
    if batch_judging:
        from evalgrid.batch_judge import extract_batchable_metrics
        batchable_metrics = extract_batchable_metrics(metric_names)
        # Only worth batching if there are 2+ batchable metrics
        if len(batchable_metrics) < 2:
            batchable_metrics = []

    async def _process_one(index: int, case: TestCase) -> Tuple[int, CaseResult]:
        async with semaphore:
            result = await asyncio.to_thread(
                _process_case_sync, case, metric_names, threshold,
                score_cache, cost_tracker, batchable_metrics,
            )
            return index, result

    if fail_fast:
        results = await _run_fail_fast(test_cases, _process_one, show_progress, len(metric_names))
    else:
        results = await _run_all(test_cases, _process_one, show_progress, len(metric_names))

    elapsed = time.time() - start_time
    throughput = len(results) / elapsed if elapsed > 0 else 0.0

    from evalgrid.judge import get_judge
    active_judge = get_judge()
    judge_stats = active_judge.stats() if active_judge else None
    judge_model = active_judge.model if active_judge else None

    _restore_judge_state(previous_judge_state, judge)

    run = EvalRun(
        results=results,
        metrics_used=metric_names,
        threshold=threshold,
        cost=cost_tracker.summary() if cost_tracker else None,
        cache_stats=score_cache.stats() if score_cache else None,
        judge_stats=judge_stats,
        judge_model=judge_model,
        elapsed_seconds=round(elapsed, 4),
        concurrency=concurrency,
        throughput=round(throughput, 2),
    )

    if not quiet:
        print(run.summary())

    return run


def _process_case_sync(
    case: TestCase,
    metric_names: List[str],
    threshold: float,
    score_cache: Optional[ScoreCache],
    cost_tracker: Optional[CostTracker],
    batchable_metrics: List[str],
) -> CaseResult:
    """
    Run all metrics for one case synchronously. Called from a worker thread.

    When ``batchable_metrics`` is non-empty, we score every batchable LLM-judge
    rubric in a single LLM call BEFORE running individual metric functions.
    Those functions will then short-circuit on a cache hit — no second call.
    """
    actual_output = _extract_output(case)

    if batchable_metrics:
        _populate_batch_judge_cache(case, actual_output, batchable_metrics, cost_tracker)
    try:
        scores = _run_metrics_for_case(case, actual_output, metric_names, score_cache, cost_tracker)
    finally:
        if batchable_metrics:
            from evals.llm_judge import clear_batch_scores
            clear_batch_scores()

    passed, failed_metrics = _evaluate_pass_fail(scores, threshold)
    return CaseResult(
        test_id=case.id,
        input=case.input,
        output=actual_output,
        expected_output=case.expected_output,
        scores=scores,
        passed=passed,
        threshold=threshold,
        failed_metrics=failed_metrics,
    )


def _populate_batch_judge_cache(
    case: TestCase,
    actual_output: str,
    batchable_metrics: List[str],
    cost_tracker: Optional[CostTracker],
) -> None:
    """
    Make ONE LLM call that scores every batchable rubric for this case, and
    install the per-rubric scores in the thread-local cache.
    """
    from evals.llm_judge import set_batch_scores
    from evalgrid.batch_judge import METRIC_TO_RUBRIC, batch_judge_score

    rubrics = [METRIC_TO_RUBRIC[m] for m in batchable_metrics]
    rubric_scores = batch_judge_score(case, actual_output, rubrics)
    set_batch_scores(rubric_scores)


async def _run_all(test_cases, process_one, show_progress: bool, num_metrics: int) -> List[CaseResult]:
    """Run every case to completion; preserve original ordering in the result list."""
    if not test_cases:
        return []
    tasks = [asyncio.create_task(process_one(i, case)) for i, case in enumerate(test_cases)]
    results_by_idx: Dict[int, CaseResult] = {}
    completed = 0
    total = len(tasks)
    description = f"Evaluating ({num_metrics} metrics)"
    _progress_init(description, total, show_progress)

    for fut in asyncio.as_completed(tasks):
        index, case_result = await fut
        results_by_idx[index] = case_result
        completed += 1
        _progress_update(description, completed, total, show_progress)

    _progress_finish(description, completed, total, show_progress)
    return [results_by_idx[i] for i in range(total)]


async def _run_fail_fast(test_cases, process_one, show_progress: bool, num_metrics: int) -> List[CaseResult]:
    """Run every case but cancel remaining tasks as soon as one fails."""
    if not test_cases:
        return []
    tasks = {asyncio.create_task(process_one(i, case)): i for i, case in enumerate(test_cases)}
    results_by_idx: Dict[int, CaseResult] = {}
    completed = 0
    total = len(tasks)
    description = f"Evaluating ({num_metrics} metrics)"
    _progress_init(description, total, show_progress)

    pending = set(tasks.keys())
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index, case_result = await task
                results_by_idx[index] = case_result
                completed += 1
                _progress_update(description, completed, total, show_progress)
                if not case_result.passed:
                    for t in pending:
                        t.cancel()
                    pending = set()
                    break
    finally:
        _progress_finish(description, completed, total, show_progress)

    return [results_by_idx[i] for i in sorted(results_by_idx)]


# ============================================================================
# RUN SAFELY FROM SYNC OR ASYNC CONTEXTS
# ============================================================================

def _run_async_safe(coro):
    """
    Run an async coroutine from sync code, even when an event loop is already running.

    Plain asyncio.run() raises if called inside an existing loop (Jupyter, FastAPI
    handlers, pytest-asyncio). To stay friendly, we detect that case and execute
    the coroutine in a dedicated worker thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(coro)

    # We're inside a running loop — hand off to a fresh loop in a worker thread.
    holder: List[Any] = []
    error: List[BaseException] = []

    def _worker():
        try:
            holder.append(asyncio.run(coro))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return holder[0]


# ============================================================================
# LIGHT-WEIGHT PROGRESS REPORTING (works under async)
# ============================================================================

_PROGRESS_STATE: Dict[str, Any] = {"last_total": 0}


def _progress_init(description: str, total: int, show: bool) -> None:
    if not show or total == 0:
        return
    _PROGRESS_STATE["start"] = time.time()


def _progress_update(description: str, completed: int, total: int, show: bool) -> None:
    if not show or total == 0:
        return
    bar_width = 30
    ratio = completed / total
    filled = int(bar_width * ratio)
    bar = "█" * filled + "░" * (bar_width - filled)
    elapsed = time.time() - _PROGRESS_STATE.get("start", time.time())
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    sys.stderr.write(
        f"\r{description}: [{bar}] {completed}/{total} ({ratio:.0%}) "
        f"· {rate:.1f}/s · ETA {eta:.1f}s    "
    )
    sys.stderr.flush()


def _progress_finish(description: str, completed: int, total: int, show: bool) -> None:
    if not show or total == 0:
        return
    elapsed = time.time() - _PROGRESS_STATE.get("start", time.time())
    rate = completed / elapsed if elapsed > 0 else 0
    sys.stderr.write(
        f"\r{description}: {completed}/{total} done in {elapsed:.2f}s "
        f"({rate:.1f} cases/s)                    \n"
    )
    sys.stderr.flush()


def quick_eval(
    input: str,
    output: str,
    expected: Optional[str] = None,
    context: Optional[str] = None,
    metrics: Union[str, List[MetricLike]] = "generation",
    threshold: float = 0.5,
) -> CaseResult:
    """
    Evaluate a single (input, output) pair in one call — the simplest possible API.

    Example:
        from evalgrid import quick_eval

        result = quick_eval(
            input="What is gravity?",
            output="Gravity is a fundamental force that attracts objects.",
            expected="Gravity is a force that attracts objects to each other.",
        )
        print(result.scores, result.passed)
    """
    case = TestCase(
        id="quick_eval",
        project="quick",
        capability="generation",
        input=input,
        expected_output=expected,
        context=context,
    )
    run = evaluate(
        cases=[(case, output)],
        metrics=metrics,
        threshold=threshold,
        progress=False,
        quiet=True,
    )
    return run.results[0]


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _coerce_cases(cases) -> List[TestCase]:
    """Accept TestCase, dict, file path, or list of those — return List[TestCase]."""
    if isinstance(cases, str):
        from loaders.dataset_loader import load_dataset
        return load_dataset(cases)

    if not isinstance(cases, (list, tuple)):
        cases = [cases]

    coerced: List[TestCase] = []
    for i, item in enumerate(cases):
        if isinstance(item, TestCase):
            coerced.append(item)
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], TestCase):
            # (case, output) pre-bundled: stash output on the case
            case = item[0]
            case.__pydantic_extra__ = getattr(case, "__pydantic_extra__", None) or {}
            setattr(case, "_actual_output", item[1])
            coerced.append(case)
        elif isinstance(item, dict):
            coerced.append(_dict_to_test_case(item, i))
        else:
            raise TypeError(
                f"Cannot interpret case #{i} as a TestCase. "
                f"Got {type(item).__name__}. Pass a TestCase, dict, or file path."
            )
    return coerced


def _dict_to_test_case(data: dict, index: int) -> TestCase:
    """Build a TestCase from a permissive dict (uses aliases for missing fields)."""
    payload = {
        "id":              str(data.get("id", f"case_{index + 1}")),
        "project":         data.get("project", "evalgrid"),
        "capability":      data.get("capability", "generation"),
        "input":           data.get("input") or data.get("question") or data.get("prompt") or "",
        "context":         data.get("context") or data.get("documents"),
        "expected_output": data.get("expected_output") or data.get("expected") or data.get("answer") or data.get("reference"),
        "expected_json":   data.get("expected_json"),
        "expected_behavior": data.get("expected_behavior"),
        "system_prompt":   data.get("system_prompt"),
        "thresholds":      data.get("thresholds", {}),
    }
    case = TestCase(**{k: v for k, v in payload.items() if v is not None or k in {"input", "id", "project", "capability"}})
    if "output" in data or "actual_output" in data:
        setattr(case, "_actual_output", data.get("output") or data.get("actual_output"))
    return case


def _extract_output(case: TestCase) -> str:
    """Pull an actual_output stash off the case (set by tuple/dict inputs)."""
    return getattr(case, "_actual_output", "") or ""


def _resolve_metric_list(metrics) -> List[str]:
    """Normalise the metrics= parameter to a list of metric name strings."""
    if isinstance(metrics, str):
        # Could be a preset name OR a single metric name
        try:
            resolved = resolve_preset(metrics)
            if resolved == MetricSet.ALL:
                return MetricRegistry.list_metrics()
            return resolved
        except ValueError:
            return [metrics]

    if metrics is MetricSet.ALL or metrics == MetricSet.ALL:
        return MetricRegistry.list_metrics()

    if isinstance(metrics, (list, tuple)):
        resolved: List[str] = []
        for m in metrics:
            if isinstance(m, str):
                resolved.append(m)
            elif callable(m):
                resolved.append(getattr(m, "__name__", str(m)))
            elif hasattr(m, "name"):
                resolved.append(m.name)
            else:
                raise TypeError(f"Cannot interpret metric: {m!r}")
        return resolved

    raise TypeError(f"metrics= must be a string, list, or MetricSet — got {type(metrics).__name__}")


def _resolve_cache(cache) -> Optional[ScoreCache]:
    if cache is False or cache is None:
        return None
    if cache is True:
        return ScoreCache()
    if isinstance(cache, ScoreCache):
        return cache
    raise TypeError(f"cache= must be bool or ScoreCache — got {type(cache).__name__}")


def _run_metrics_for_case(
    case: TestCase,
    actual_output: str,
    metric_names: List[str],
    score_cache: Optional[ScoreCache],
    cost_tracker: Optional[CostTracker],
) -> Dict[str, float]:
    """Run each metric in turn, honouring cache + cost tracker. Never raises."""
    scores: Dict[str, float] = {}
    for metric_name in metric_names:
        # Cache lookup
        if score_cache:
            cached = score_cache.get(metric_name, case, actual_output)
            if cached is not None:
                scores.update(cached)
                continue

        result = _safe_compute_metric(metric_name, case, actual_output)
        if result is None:
            continue
        if isinstance(result, dict):
            scores.update(result)
            if score_cache:
                score_cache.put(metric_name, case, actual_output, result)
        else:
            scores[metric_name] = float(result)
            if score_cache:
                score_cache.put(metric_name, case, actual_output, {metric_name: float(result)})

        if cost_tracker:
            # Best-effort cost tracking — judges run one LLM call per metric per case
            if metric_name.startswith("llm_judge") or metric_name in {"refusal_quality", "behavior_correctness", "summarization_quality", "prompt_alignment"}:
                cost_tracker.record(metric_name, input_text=case.input, output_text=actual_output)

    return scores


def _safe_compute_metric(metric_name: str, case: TestCase, output: str) -> Any:
    """Compute one metric with friendly error swallowing."""
    try:
        # Class-based metric path
        registry = MetricRegistry()
        if metric_name in registry._metrics:
            score = registry._metrics[metric_name].compute(case, output)
            return {metric_name: float(score)} if not isinstance(score, dict) else score
        if metric_name in registry._metric_functions:
            result = registry._metric_functions[metric_name]["func"](case, output)
            return result
        return None
    except Exception:
        return None


def _evaluate_pass_fail(scores: Dict[str, float], threshold: float):
    """Decide pass/fail; return (passed, failed_metric_names)."""
    failed: List[str] = []
    for name, value in scores.items():
        try:
            if float(value) < threshold:
                failed.append(name)
        except (TypeError, ValueError):
            continue
    return (len(failed) == 0, failed)


def _uses_llm_judge(metric_names: List[str]) -> bool:
    """True when any of the requested metrics actually hits the LLM judge."""
    llm_metric_prefixes = ("llm_judge", "summarization_quality", "summarization_faithfulness",
                          "prompt_alignment", "behavior_correctness", "refusal_quality")
    return any(m.startswith(llm_metric_prefixes) or m == "judge_correctness" or m == "judge_groundedness"
               for m in metric_names)


def _configure_judge_for_run(
    judge: Any,
    metric_names: List[str],
    cost_tracker: Optional[CostTracker],
) -> Dict[str, Any]:
    """
    Set up the global LLM judge for this evaluation run.

    Returns a snapshot of the previous judge state so the caller can restore it
    when the run ends. The contract:

      judge=False         → force heuristic mode (clear any existing judge for this run)
      judge=None          → auto-detect on first use (if metrics need it)
      judge="model-name"  → set this specific model
      judge=adapter/JC    → use the given client directly
    """
    from evals.llm_judge import get_llm_client, set_llm_client
    from evalgrid.judge import (
        JudgeClient,
        configure as judge_configure,
        ensure_judge_configured,
        set_judge,
    )

    previous = {"client": get_llm_client(), "changed": False}

    if judge is False:
        set_llm_client(None)
        previous["changed"] = True
    elif judge is None:
        if _uses_llm_judge(metric_names):
            client = ensure_judge_configured()
            if client is not None and cost_tracker is not None:
                # Bind cost tracker so we record every REAL LLM call.
                client.cost_tracker = cost_tracker
    elif isinstance(judge, str):
        client = judge_configure(judge=judge)
        if client is not None and cost_tracker is not None:
            client.cost_tracker = cost_tracker
        previous["changed"] = True
    else:
        client = set_judge(judge)
        if isinstance(client, JudgeClient) and cost_tracker is not None:
            client.cost_tracker = cost_tracker
        previous["changed"] = True

    return previous


def _restore_judge_state(previous: Dict[str, Any], judge_param: Any) -> None:
    """
    Restore the previous judge state after a run.

    Only restores when this evaluate() call modified the judge (judge != None).
    When judge=None and a judge was auto-detected, leave it set so subsequent
    evaluate() calls don't re-detect.
    """
    if not previous.get("changed"):
        return
    from evals.llm_judge import set_llm_client
    set_llm_client(previous.get("client"))
