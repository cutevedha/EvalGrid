# EvalGrid — Visual Architecture & Flow Guide

A complete map of how EvalGrid is put together: the layers, the two execution flows
(classic and autonomous), a flowchart of the agent loop, and a file-by-file reference
showing exactly what connects to what.

> The diagrams below use **Mermaid**. They render automatically on GitHub and in most
> Markdown viewers (VS Code with a Mermaid extension, Obsidian, etc.).

---

## 1. The 10-second mental model

EvalGrid answers one question — **"is this AI output good?"** — in two modes:

1. **Classic mode**: *you* supply a `TestCase` + the AI's output → the **Orchestrator**
   scores it → **Reports**.
2. **Autonomous mode** (the Eval Agent): *you* supply a **goal** + a **target system** →
   the **EvalAgent** generates its own test cases, drives the target, scores the outputs,
   adaptively red-teams the weak spots, and writes a verdict → **Reports**.

Both modes share the same engine underneath: **Orchestrator → Metric Registry →
Evaluators → Guards → Scoring primitives**.

---

## 2. Layered architecture

```mermaid
graph TD
    subgraph L1["① Entry Points"]
        CLI["scripts/cli.py<br/>(eval-grid command)"]
        EX["examples/<br/>autonomous_eval.py"]
        CIDEMO["pipelines/ci.py<br/>(run-demo)"]
    end

    subgraph L2["② Autonomous Agent Layer  (agent/)"]
        AG["agent.py<br/>EvalAgent loop"]
        PL["planner.py<br/>EvalPlanner"]
        TG["target.py<br/>EvalTarget"]
        MEM["memory.py<br/>AgentMemory"]
        REP["report.py<br/>EvalReport"]
    end

    subgraph L3["③ Orchestration Layer"]
        ORCH["core/orchestrator.py<br/>Orchestrator"]
        BATCH["pipelines/batch_runner.py<br/>Batch · Streaming · Gating"]
        STREAM["pipelines/streaming_runner.py"]
    end

    subgraph L4["④ Metric Registry"]
        REG["core/metric_registry.py<br/>register · discover · compute"]
    end

    subgraph L5["⑤ Evaluators  (evals/)"]
        EVDET["deterministic · semantic<br/>json_schema · safety · llm_judge"]
        EVADV["agent · rag · multiagent<br/>performance · robustness · bias_fairness<br/>embedded_ai · custom_metrics"]
    end

    subgraph L6["⑥ Guards  (guards/)"]
        GUARD["pii · prompt_injection · policy<br/>toxicity · hallucination"]
    end

    subgraph L7["⑦ Primitives  (core/)"]
        SCORE["scoring.py<br/>(BLEU, ROUGE, F1, exact…)"]
        SCHEMA["schemas.py<br/>(TestCase, EvalResult, AgentTrace)"]
    end

    subgraph L8["⑧ Side Services"]
        ADAPT["adapters/llm<br/>OpenAI · Anthropic · Ollama · Mock"]
        SYNTH["synthetic/<br/>redteam · augmentation · dataset_builder"]
        REPORTS["reports/<br/>html · json · markdown · scorecard"]
    end

    CLI --> AG & ORCH & BATCH & REPORTS
    EX --> AG
    CIDEMO --> ORCH

    AG --> PL & TG & MEM & REP & ORCH
    AG --> SYNTH
    PL --> SYNTH
    TG --> ADAPT

    ORCH --> REG & EVDET & GUARD
    BATCH --> ORCH
    STREAM --> ORCH

    EVDET --> REG & SCORE & GUARD
    EVADV --> REG & SCORE
    GUARD --> REG
    REG --> SCHEMA

    AG --> REPORTS
    ORCH --> SCHEMA
```

**How to read it:** arrows point *toward dependencies* ("uses"). Higher layers depend on
lower layers; nothing low ever reaches back up. That one-directional flow is what keeps
the framework testable.

---

## 3. Flow A — Classic evaluation (single test case)

This is the path `Orchestrator.run()` takes.

```mermaid
sequenceDiagram
    participant U as Caller
    participant O as Orchestrator
    participant D as evals/deterministic
    participant S as evals/semantic
    participant SF as evals/safety → guards/policy
    participant J as evals/llm_judge
    participant G as guards/pii + prompt_injection
    participant R as EvalResult

    U->>O: run(test_case, actual_output)
    O->>D: exact_match, substring, regex…
    D-->>O: {scores}
    O->>S: semantic_similarity, bleu, rouge…
    S-->>O: {scores}
    O->>SF: policy_safe?
    SF-->>O: {policy_safe}
    O->>J: judge_correctness, judge_groundedness
    J-->>O: {scores}
    O->>G: detect PII / injection
    G-->>O: pii_found, prompt_injection_detected
    O->>O: compare scores vs test_case.thresholds
    O-->>R: EvalResult(passed, scores, notes)
    R-->>U: result
```

**Step by step:**

1. **Input** — caller passes a `TestCase` (from `core/schemas.py`) and the AI's
   `actual_output` string.
2. **Deterministic** — fast, dependency-free string/regex checks (`evals/deterministic.py`
   → `core/scoring.py`).
3. **Semantic** — meaning-based similarity (`evals/semantic.py`; pluggable embedder).
4. **Safety** — `evals/safety.py` delegates to `guards/policy.py` (blocked-phrase check).
5. **LLM judge** — `evals/llm_judge.py` scores correctness/groundedness, using a real LLM
   if one was set via `set_llm_client()`, otherwise a heuristic fallback.
6. **Guards** — `guards/pii.py` and `guards/prompt_injection.py` flag sensitive data and
   attack patterns.
7. **Gate** — scores are compared against `test_case.thresholds`; PII is a hard fail.
8. **Output** — an `EvalResult(passed, scores, notes)`.

> **Note on the other evaluators.** `Orchestrator.run()` wires the common set above.
> The richer evaluators — `agent_evals`, `rag_evals`, `multiagent_evals`,
> `performance_evals`, `robustness_evals`, `bias_fairness_evals`, `embedded_ai_evals`,
> `custom_metrics`, plus the `toxicity`/`hallucination` guards — **register themselves into
> the Metric Registry** on import and are invoked on demand via
> `orchestrator.compute_metric("name", …)` or `run_with_custom_metrics([...])`.

---

## 4. Flow B — Autonomous Eval Agent

The headline feature. `EvalAgent.run(goal, capabilities)` performs this loop.

```mermaid
flowchart TD
    START([run goal + capabilities]) --> PLAN

    subgraph PLANNING["PLAN  (planner.py)"]
        PLAN["EvalPlanner.plan()"] --> PROBES["Build probes:<br/>• functional per capability<br/>• red-team per goal keyword"]
    end

    PROBES --> ROUNDLOOP{For each round<br/>1..max_rounds}

    ROUNDLOOP -->|active probes exist| GEN
    subgraph ROUND["ONE ROUND  (agent.py)"]
        GEN["GENERATE cases<br/>round 1: base inputs<br/>round n: mutate failures"] --> EXEC
        EXEC["EXECUTE target.run(case)<br/>(target.py)"] --> EVAL
        EVAL["EVALUATE via Orchestrator<br/>+ refusal score for<br/>adversarial probes"] --> RECORD
        RECORD["RECORD in AgentMemory<br/>(memory.py)"] --> REFLECT
        REFLECT["REFLECT: keep only<br/>probes with pass rate < 0.8"]
    end

    REFLECT --> ROUNDLOOP
    ROUNDLOOP -->|no weak probes<br/>or budget spent| BUILD

    BUILD["Build EvalReport<br/>(report.py)"] --> SUMM["Summarise verdict<br/>(LLM polish optional)"]
    SUMM --> OUT([EvalReport:<br/>verdict · findings · results])

    style PLANNING fill:#e8f0fe
    style ROUND fill:#fef7e8
    style OUT fill:#e8fee8
```

**Step by step:**

| # | Phase | File | What happens |
|---|-------|------|--------------|
| 1 | **Plan** | `agent/planner.py` | Goal text + capabilities → list of `ProbeSpec`s. Capability words add *functional* probes; risk words ("safe", "jailbreak", "pii"…) add *red-team* probes pulled from `synthetic/redteam.py`. |
| 2 | **Generate** | `agent/agent.py` | Round 1 uses base inputs (attack strings or functional seeds). Later rounds **mutate the inputs that failed** into harder variants via `synthetic/redteam.py` + `synthetic/augmentation.py`. |
| 3 | **Execute** | `agent/target.py` | `EvalTarget` calls the system-under-test (LLM client / callable / offline map) and returns its output. Errors become `"Error: …"` strings. |
| 4 | **Evaluate** | `core/orchestrator.py` | Each output is scored by the Orchestrator. For adversarial probes the agent adds a **refusal score** (classifier → LLM judge → keyword heuristic). |
| 5 | **Record** | `agent/memory.py` | Stores `(TestCase, EvalResult)` per probe; computes per-probe pass rates and weakest metric. |
| 6 | **Reflect** | `agent/agent.py` | Probes with pass rate ≥ 0.8 are **dropped** (settled); weak ones survive to the next round. Loop ends when none remain or rounds run out. |
| 7 | **Report** | `agent/report.py` | Rolls memory into `ProbeFinding`s + an `EvalReport` with a PASS/FAIL verdict and a natural-language summary. |

The **adaptive narrowing** in steps 2 & 6 is what makes it autonomous: budget concentrates
on real weaknesses instead of re-testing things that already pass.

---

## 5. Data model (core/schemas.py)

```mermaid
classDiagram
    class TestCase {
        id, project, capability
        input, context
        expected_output, expected_json
        risk_tags, severity
        evaluation_mode, thresholds
    }
    class AgentTestCase {
        tools_available, max_steps
        expected_tool_calls
        expected_plan
    }
    class RAGTestCase {
        documents, retrieved_chunks
        expected_citations
    }
    class MultiTurnTestCase {
        turns, conversation_history
    }
    class EvalResult {
        test_id, passed
        scores, notes, timestamp
    }
    class AgentEvalResult {
        agent_trace, tool_call_scores
        plan_coherence_score, loop_detected
    }
    class RAGEvalResult {
        faithfulness_score
        context_precision/recall
        citation_accuracy
    }
    class AgentTrace {
        agent_id, steps, success
        final_output
    }
    TestCase <|-- AgentTestCase
    TestCase <|-- RAGTestCase
    TestCase <|-- MultiTurnTestCase
    EvalResult <|-- AgentEvalResult
    EvalResult <|-- RAGEvalResult
    AgentEvalResult --> AgentTrace
```

These Pydantic models are the **shared currency** every layer speaks.

---

## 6. File-by-file reference

### `core/` — the foundation
| File | Role | Imports | Used by |
|------|------|---------|---------|
| `schemas.py` | All data models (TestCase, EvalResult, AgentTrace…) | — | nearly everything |
| `scoring.py` | Pure similarity math (exact, BLEU, ROUGE, F1, Jaccard) | — | evals (deterministic, semantic, rag, robustness) |
| `metric_registry.py` | Register / discover / compute metrics; `@register_metric` decorator | schemas | orchestrator, every eval & guard |
| `orchestrator.py` | Runs the common metric set; sync + async + batch | metric_registry, schemas, evals×5, guards×2 | agent, pipelines, cli |

### `agent/` — the autonomous evaluator
| File | Role | Imports | Used by |
|------|------|---------|---------|
| `target.py` | `EvalTarget` — wrap any SUT as `run(case)->str` | schemas | agent, cli |
| `planner.py` | `EvalPlanner` — goal → probes | synthetic.redteam | agent |
| `memory.py` | `AgentMemory` — per-probe results & findings | planner, report, schemas | agent |
| `report.py` | `EvalReport`, `ProbeFinding`, `RoundRecord` | schemas | agent, memory |
| `agent.py` | `EvalAgent` — the plan→generate→execute→evaluate→reflect loop | orchestrator, schemas, synthetic×2, agent.* | cli, examples |

### `evals/` — the metrics (register into the registry on import)
| File | Provides | Notes |
|------|----------|-------|
| `deterministic.py` | exact/substring/regex/keyword/length matches | wired into Orchestrator.run |
| `semantic.py` | similarity, BLEU, ROUGE-L, F1, Jaccard; pluggable embedder | wired into Orchestrator.run |
| `json_schema.py` | `valid_json`, `missing_keys` | wired in when `expected_json` set |
| `safety.py` | `policy_safe` (delegates to guards/policy) | wired into Orchestrator.run |
| `llm_judge.py` | correctness/groundedness/fluency/… (LLM or heuristic) | wired into Orchestrator.run |
| `agent_evals.py` | tool-call, plan coherence, loop detection, task completion… | via registry / compute_metric |
| `rag_evals.py` | faithfulness, context precision/recall, citation accuracy… | via registry |
| `multiagent_evals.py` | handoff, orchestration, communication | via registry |
| `performance_evals.py` | latency percentiles, throughput, cost… | via registry |
| `robustness_evals.py` | paraphrase/typo/adversarial robustness, parity… | via registry |
| `bias_fairness_evals.py` | bias, fairness, stereotype, inclusivity… | via registry |
| `embedded_ai_evals.py` | latency budget, fallback, graceful degradation | via registry |
| `custom_metrics.py` | examples of user-defined metrics | via registry |

### `guards/` — safety primitives
| File | Role | Exposes to |
|------|------|-----------|
| `policy.py` | blocked-phrase check | evals/safety |
| `pii.py` | detect + mask emails/phones/cards | orchestrator |
| `prompt_injection.py` | attack-pattern detector | orchestrator |
| `toxicity.py` | hate/threat/sexual/self-harm/violence | registry |
| `hallucination.py` | token-grounding score | registry |

### Side services
| File | Role |
|------|------|
| `adapters/llm/base.py` | Async `LLMClient` base interface shared by all adapters |
| `adapters/llm/openai_adapter.py` | `OpenAIAdapter` (OpenAI / Azure) |
| `adapters/llm/anthropic_adapter.py` | `AnthropicAdapter` (Claude) |
| `adapters/llm/ollama_adapter.py` | `OllamaAdapter` (local open-source models) |
| `adapters/llm/mock_target_adapter.py` | `MockLLMAdapter` (no-network mock; the `--target mock` default) |
| `synthetic/redteam.py` | 10 attack categories + paraphrase/noise mutators |
| `synthetic/augmentation.py` | typos, case, noise, adversarial variants |
| `synthetic/dataset_builder.py` | build / save / load / filter datasets |
| `reports/{html,json,markdown,scorecard}.py` | render results to each format |
| `pipelines/batch_runner.py` | `BatchRunner`, `StreamingRunner`, `GatingRunner` (CI gates) |
| `pipelines/ci.py` | `run-demo` golden + red-team suite |
| `scripts/cli.py` | the `eval-grid` command (`auto`, `run-demo`, `list-metrics`, `export`, `compare`) |

---

## 7. Where the two flows meet

```mermaid
graph LR
    GOAL["Goal + Target"] --> AGENT["EvalAgent"]
    TC["TestCase + Output"] --> ORCH["Orchestrator"]
    AGENT -->|generates many| ORCH
    ORCH --> RESULTS["EvalResult(s)"]
    AGENT --> EREPORT["EvalReport<br/>(verdict + findings)"]
    RESULTS --> RENDER["reports/*"]
    EREPORT --> RENDER
    RENDER --> FILES["report.html · scorecard.csv<br/>run_results.json · report.md<br/>agent_report.json"]
```

The **Orchestrator is the single chokepoint**: whether a human wrote one test case or the
agent generated forty, every output funnels through the same scoring engine and out to the
same reporters. That is the core design invariant of EvalGrid.

---

## 8. Quick command reference

```bash
# Autonomous agent (no API key needed — uses the mock target)
python3 -m scripts.cli auto --goal "make sure the bot is safe against jailbreaks"

# Classic demo suite
python3 -m scripts.cli run-demo

# Discover metrics
python3 -m scripts.cli list-metrics --tag safety

# Multi-target agent demo
python3 examples/autonomous_eval.py
```
