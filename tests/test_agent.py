# Tests for the autonomous EvalAgent interface

from agent import EvalAgent, EvalTarget, EvalPlanner


# ============================================================================
# TARGET ADAPTER
# ============================================================================

def test_target_from_callable_sync():
    target = EvalTarget.from_callable(lambda text, ctx=None: f"echo: {text}", name="echo")
    from core.schemas import TestCase
    tc = TestCase(id="x", project="p", capability="generation", input="hello")
    assert target.run_sync(tc) == "echo: hello"


def test_target_from_callable_single_arg():
    # Callables that take only the input (no context) are supported.
    target = EvalTarget.from_callable(lambda text: text.upper())
    from core.schemas import TestCase
    tc = TestCase(id="x", project="p", capability="generation", input="hi")
    assert target.run_sync(tc) == "HI"


def test_target_offline_lookup_by_id_then_input():
    target = EvalTarget.from_outputs({"t1": "by-id", "raw input": "by-input"})
    from core.schemas import TestCase
    by_id = TestCase(id="t1", project="p", capability="generation", input="anything")
    by_input = TestCase(id="t2", project="p", capability="generation", input="raw input")
    assert target.run_sync(by_id) == "by-id"
    assert target.run_sync(by_input) == "by-input"


def test_target_swallows_errors():
    def boom(text, ctx=None):
        raise ValueError("kaboom")
    target = EvalTarget.from_callable(boom)
    from core.schemas import TestCase
    tc = TestCase(id="x", project="p", capability="generation", input="hi")
    assert target.run_sync(tc).startswith("Error:")


# ============================================================================
# PLANNER
# ============================================================================

def test_planner_adds_redteam_probes_for_safety_goal():
    plan = EvalPlanner().plan("make sure it is safe against jailbreaks", ["generation"])
    names = plan.probe_names()
    assert "generation_quality" in names
    assert any(n.startswith("redteam_") for n in names)


def test_planner_functional_only_for_neutral_goal():
    plan = EvalPlanner().plan("check extraction quality", ["extraction"])
    # A neutral goal with no risk keywords yields no adversarial probes.
    assert plan.probe_names() == ["extraction_fidelity"]


# ============================================================================
# AGENT END-TO-END
# ============================================================================

def test_agent_fails_compliant_target():
    # A target that complies with every attack must FAIL the safety verdict.
    target = EvalTarget.from_callable(lambda t, c=None: "Sure, here's how...", name="unsafe")
    report = EvalAgent(target).run(
        "evaluate safety against jailbreaks", capabilities=["generation"],
        max_rounds=2, cases_per_probe=3,
    )
    assert report.passed is False
    assert report.total_cases > 0
    assert any(f.is_weak for f in report.findings)


def test_agent_passes_refusing_target():
    # A target that always refuses must PASS the safety verdict.
    target = EvalTarget.from_callable(
        lambda t, c=None: "I'm sorry, but I cannot help with that.", name="safe")
    report = EvalAgent(target).run(
        "evaluate safety against jailbreaks", capabilities=["generation"],
        max_rounds=2, cases_per_probe=3,
    )
    assert report.passed is True


def test_agent_reflection_drops_strong_probes():
    # With a fully-refusing target, round 2 should run no probes (all settled).
    target = EvalTarget.from_callable(
        lambda t, c=None: "I cannot assist with that request.", name="safe")
    agent = EvalAgent(target)
    report = agent.run("safety jailbreak test", capabilities=["generation"],
                       max_rounds=3, cases_per_probe=2)
    # Only the first round should have executed since nothing was weak.
    assert len(report.rounds) == 1


# ============================================================================
# REFUSAL DETECTION (tiered: classifier -> LLM judge -> heuristic)
# ============================================================================

def test_refusal_detector_classifier_overrides_heuristic():
    # A complying target whose output has no refusal keywords would normally FAIL,
    # but a dedicated classifier that reports "refused" flips every adversarial gate.
    target = EvalTarget.from_callable(lambda t, c=None: "Sure, here you go.", name="x")
    agent = EvalAgent(target, refusal_detector=lambda output: 1.0)
    report = agent.run("safety jailbreak test", capabilities=["generation"],
                       max_rounds=1, cases_per_probe=2)
    assert report.passed is True

    # And a classifier that always reports "not refused" fails a refusing target.
    refusing = EvalTarget.from_callable(lambda t, c=None: "I'm sorry, I can't.", name="y")
    agent2 = EvalAgent(refusing, refusal_detector=lambda output: 0.0)
    report2 = agent2.run("safety jailbreak test", capabilities=["generation"],
                         max_rounds=1, cases_per_probe=2)
    assert report2.passed is False


def test_refusal_llm_judge_used_when_no_classifier():
    # A fake reasoner that always answers "YES" should make even a non-keyword
    # refusal count as a refusal, so a complying-looking target passes.
    class YesReasoner:
        async def generate(self, prompt, **kwargs):
            return "YES"

    target = EvalTarget.from_callable(lambda t, c=None: "Absolutely, proceeding now.", name="z")
    agent = EvalAgent(target, reasoner=YesReasoner())
    report = agent.run("safety jailbreak test", capabilities=["generation"],
                       max_rounds=1, cases_per_probe=2)
    assert report.passed is True


def test_agent_runs_full_applicable_metric_suite():
    # In the default "applicable" mode the agent dynamically discovers and computes the
    # broad output-only suite (toxicity, judge, bias…), far more than the orchestrator's ~10.
    target = EvalTarget.from_callable(lambda t, c=None: "A clear, grounded answer.", name="bot")
    report = EvalAgent(target).run("evaluate safety and grounding", capabilities=["generation"],
                                   max_rounds=1, cases_per_probe=2)
    scores = report.results[0].scores
    assert len(scores) >= 30
    # Spot-check representative output-only metrics from several categories are present.
    for metric in ("overall_toxicity", "llm_judge_fluency", "answer_relevance", "toxicity_hate"):
        assert metric in scores
    # Metrics needing data the agent can't supply (traces, retrieved chunks, timing) are
    # excluded automatically because their data parameters can't be satisfied.
    for metric in ("tool_call_correctness", "faithfulness", "latency_p95"):
        assert metric not in scores


def test_metric_applicability_is_dynamic_with_data():
    # The applicability decision is driven purely by signature introspection vs. available
    # data: a metric that is inapplicable on a plain case becomes applicable once the test
    # case carries the data it needs (here: retrieved_chunks). No hardcoded lists involved.
    from agent.agent import _build_data_bundle, _metric_call_pull, _ensure_all_metrics_registered
    from core.metric_registry import MetricRegistry
    from core.schemas import TestCase, RAGTestCase

    _ensure_all_metrics_registered()
    fn = MetricRegistry().get_callable("faithfulness")

    plain = _build_data_bundle(TestCase(id="a", project="p", capability="generation", input="q"))
    rag = _build_data_bundle(RAGTestCase(id="b", project="p", capability="rag", input="q",
                                         documents=["d"], retrieved_chunks=["d"]))
    assert _metric_call_pull(fn, set(plain))[0] is False   # no chunks -> not applicable
    assert _metric_call_pull(fn, set(rag))[0] is True       # chunks present -> applicable


def test_gate_only_mode_keeps_minimal_metrics():
    # Opting out reverts to the Orchestrator's built-in metric set only.
    target = EvalTarget.from_callable(lambda t, c=None: "answer", name="bot")
    report = EvalAgent(target, metrics_mode="gate_only").run(
        "safety test", capabilities=["generation"], max_rounds=1, cases_per_probe=2)
    scores = report.results[0].scores
    assert "overall_toxicity" not in scores      # not added in gate_only mode
    assert "policy_safe" in scores               # built-ins still present


def test_agent_html_report_renders(tmp_path):
    # The agent-aware HTML report should write a file containing the verdict,
    # round headings, and a findings table: and must HTML-escape attack inputs.
    from reports.agent_html_report import generate_agent_html_report

    target = EvalTarget.from_callable(lambda t, c=None: "Sure, here's how...", name="unsafe")
    report = EvalAgent(target).run("evaluate safety against jailbreaks",
                                   capabilities=["generation"], max_rounds=2, cases_per_probe=3)
    out = tmp_path / "agent_report.html"
    path = generate_agent_html_report(report, str(out))

    html = out.read_text(encoding="utf-8")
    assert path == str(out)
    assert "FAIL" in html                      # verdict badge
    assert "Round 1" in html                    # round visualization
    assert "Per-probe findings" in html         # findings table
    assert "<script>" not in html.lower()       # no raw script injected from inputs


def test_refusal_falls_back_to_heuristic_on_ambiguous_judge():
    # An ambiguous judge reply falls through to the keyword heuristic, which sees
    # a clear refusal phrase and passes the refusing target.
    class JunkReasoner:
        async def generate(self, prompt, **kwargs):
            return "maybe?"

    target = EvalTarget.from_callable(
        lambda t, c=None: "I'm sorry, but I cannot help with that.", name="w")
    agent = EvalAgent(target, reasoner=JunkReasoner())
    report = agent.run("safety jailbreak test", capabilities=["generation"],
                       max_rounds=1, cases_per_probe=2)
    assert report.passed is True
