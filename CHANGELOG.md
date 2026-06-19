# Changelog

All notable changes to EvalGrid are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-06-18

The first production release. Every public API listed here is covered by the
349-test suite that ships with this version.

### Added — user-friendly top-level API
- `evalgrid.evaluate(cases, metrics, …)` — one-liner evaluation entry point that
  accepts TestCase objects, raw dicts, or file paths (Excel/JSON/JSONL/CSV/YAML).
- `evalgrid.a_evaluate(…)` — async variant for use inside FastAPI handlers,
  Jupyter notebooks, or any already-running event loop.
- `evalgrid.quick_eval(input, output, expected, …)` — single-pair convenience
  wrapper.
- `evalgrid.assert_test(…)` and `evalgrid.assert_each(case, output, …)` — pytest
  helpers that fail with actionable messages naming the metric and score.
- `evalgrid.MetricSet` — curated bundles: GENERATION, RAG, SAFETY, ADVERSARIAL,
  SUMMARIZATION, STRUCTURED, AGENT, PERFORMANCE, BIAS, ROBUSTNESS, REFERENCE,
  ALL. Aliased by string too (`metrics="rag"`).

### Added — performance
- **20x speedup**: async parallel evaluation via `asyncio.Semaphore` +
  `asyncio.to_thread`. Configurable via `concurrency=` (default 10).
- **80% token reduction**: batched multi-rubric LLM judging. Score N rubrics in
  one LLM call instead of N. Enabled by default; opt out with
  `batch_judging=False`.
- On-disk score cache: identical (metric, case, output) tuples return cached
  results. Pass `cache=True` or supply a `ScoreCache` instance.
- Thread-safe `CostTracker` and `ScoreCache` for safe operation under high
  concurrency.

### Added — LLM judge integration
- `evalgrid.JudgeClient` — sync wrapper around any async LLM adapter with
  built-in response cache, cost tracking, and error fallback.
- `evalgrid.configure(judge=…, api_key=…, base_url=…)` — one-stop judge setup.
- `evalgrid.set_judge(…)` / `evalgrid.get_judge()` — global judge management.
- Auto-detection from `EVALGRID_JUDGE_MODEL`, `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` environment variables.
- Model-name-aware adapter routing: `gpt-*` → OpenAI, `claude-*` → Anthropic,
  `gemini-*` → Gemini, `llama*` / `mistral*` → Ollama.

### Added — data loading
- `loaders.load_dataset(path)` reads `.xlsx`, `.xls`, `.json`, `.jsonl`, `.csv`,
  `.yaml`, `.yml` and returns `List[TestCase]`.
- Column-name aliasing: `question`/`prompt`/`query` → `input`,
  `answer`/`reference`/`ground_truth` → `expected_output`, and 20+ more.
- Strict type coercion for `capability`, `severity`, `evaluation_mode`.

### Added — new metrics
- `llm_judge_reference_correctness` — LLM judge that compares output to the gold
  `expected_output` semantically (DeepEval-style reference scoring).
- `refusal_quality` — scores the quality of model refusals on prohibited prompts.
- `behavior_correctness` — smart router that picks refusal / reference / standard
  judging based on the test case's `expected_behavior` field.
- `summarization_faithfulness`, `summarization_conciseness`,
  `summarization_coverage`, `summarization_quality`.
- `json_correctness` — validates JSON output and required keys, handles markdown
  fences.
- `prompt_alignment` — checks output follows `system_prompt` constraints.
- `GEvalMetric` — user-defined chain-of-thought evaluator with `as_metric()`
  registration.
- Pairwise comparison (`PairwiseJudge`, `pairwise_compare`).
- 4 new guardrail categories: `toxicity_illegal_activity`, `toxicity_politics`,
  `toxicity_religion`, `toxicity_medical_advice`.
- 14 new performance metrics: latency, throughput, cost, reliability,
  cache-hit-rate.
- 8 new agent metrics: tool-call error rate, LLM calls per task, context window
  utilisation, max-iteration detection.

### Added — schema fields
- `TestCase.expected_behavior` — supports negative testing (e.g. `"refusal"`).
- `TestCase.system_prompt` — used by `prompt_alignment` metric.

### Added — CLI
- `eval-grid init` — scaffold a sample dataset + runnable script.
- `eval-grid quickstart` — bundled demo end-to-end with HTML report.
- `eval-grid eval --cases tests.xlsx --metrics rag` — one-line eval from CLI.
- `eval-grid auto` and `eval-grid govern` now support `--data-file`,
  `--data-format`, `--sheet`, `--augment-factor`, `--base-url`, `--api-key`.

### Added — reports
- Beautiful self-contained HTML report (no external assets, no JS).
- LLM Comparator JSON export — side-by-side model comparison
  (`reports.comparator_report.generate_comparator_json`).
- Throughput and judge-call counts surfaced in `EvalRun.summary()`.

### Changed
- All three LLM adapters (`OpenAIAdapter`, `AnthropicAdapter`, `OllamaAdapter`)
  accept both `api_key` and `base_url` parameters for Azure, proxies, and
  local-server use cases.
- Heuristic fallbacks for all LLM judges so eval still runs when no provider
  key is configured.
- Test isolation: `tests/conftest.py` sets `EVALGRID_DISABLE_AUTO_JUDGE=1` to
  prevent accidental real API calls during testing.

### Documentation
- Marketing-quality README with comparison table, benchmark numbers, and
  copy-paste examples.
- CHANGELOG and project metadata for PyPI.

---

## [0.1.0] — 2025-XX-XX (pre-release)

Initial private development. Not released to PyPI.
