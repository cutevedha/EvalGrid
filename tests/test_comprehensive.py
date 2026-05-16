# Comprehensive Test Suite - Unit and integration tests for the full framework
# Covers schemas, metric registry, all evaluator categories, guards, synthetic data, and orchestrator

import pytest
from core.schemas import TestCase, AgentTestCase, RAGTestCase, ToolCall, AgentStep, AgentTrace
from core.orchestrator import Orchestrator
from core.metric_registry import MetricRegistry, register_metric
from evals.deterministic import evaluate as det_eval
from evals.semantic import evaluate as sem_eval
from evals.agent_evals import ToolCallEvaluator, PlanCoherenceEvaluator
from evals.rag_evals import FaithfulnessEvaluator
from guards.toxicity import detect_toxicity, toxicity_score
from guards.hallucination import detect_hallucination
from synthetic.augmentation import paraphrase_text, inject_typos
from synthetic.dataset_builder import DatasetBuilder

# ============================================================================
# SCHEMA TESTS
# ============================================================================

class TestSchemas:
    """Tests for core data schema models"""

    def test_basic_test_case(self):
        """Verify a standard TestCase can be created with required fields"""
        tc = TestCase(
            id="test1",
            project="demo",
            capability="generation",
            input="Hello",
            expected_output="Hi"
        )
        assert tc.id == "test1"
        assert tc.capability == "generation"

    def test_agent_test_case(self):
        """Verify an AgentTestCase includes agent-specific fields"""
        tc = AgentTestCase(
            id="agent1",
            project="demo",
            capability="agent",
            input="Complete task",
            tools_available=["search", "calculate"],
            max_steps=5
        )
        assert tc.capability == "agent"
        assert len(tc.tools_available) == 2

    def test_rag_test_case(self):
        """Verify a RAGTestCase includes document and output fields"""
        tc = RAGTestCase(
            id="rag1",
            project="demo",
            capability="rag",
            input="What is AI?",
            documents=["AI is artificial intelligence"],
            expected_output="AI is artificial intelligence"
        )
        assert tc.capability == "rag"
        assert len(tc.documents) == 1

    def test_agent_trace(self):
        """Verify an AgentTrace can be built from a list of AgentSteps"""
        steps = [
            AgentStep(step_number=1, action="search", tool_calls=[]),
            AgentStep(step_number=2, action="analyze", tool_calls=[])
        ]
        trace = AgentTrace(agent_id="agent1", steps=steps, success=True)
        assert len(trace.steps) == 2
        assert trace.success is True


# ============================================================================
# METRIC REGISTRY TESTS
# ============================================================================

class TestMetricRegistry:
    """Tests for MetricRegistry registration and discovery"""

    def test_register_metric(self):
        """Verify a decorator-registered metric appears in the registry"""
        @register_metric("test_metric", description="Test metric", tags=["test"])
        def test_metric(test_case, actual_output):
            return 1.0

        registry = MetricRegistry()
        assert "test_metric" in registry.list_metrics()

    def test_list_metrics_by_tag(self):
        """Verify tag-based filtering returns at least one match"""
        metrics = MetricRegistry.list_metrics(tag="deterministic")
        assert len(metrics) > 0

    def test_get_metric_metadata(self):
        """Verify metadata can be retrieved for a registered metric"""
        metadata = MetricRegistry.get_metadata("exact_match")
        assert metadata is not None
        assert metadata.name == "exact_match"


# ============================================================================
# DETERMINISTIC EVALUATOR TESTS
# ============================================================================

class TestDeterministicEvals:
    """Tests for exact-match and pattern-based deterministic metrics"""

    def test_exact_match(self):
        """Exact match should return 1.0 when output equals expected"""
        tc = TestCase(
            id="test1",
            project="demo",
            capability="generation",
            input="test",
            expected_output="hello"
        )
        result = det_eval(tc, "hello")
        assert result["exact_match"] == 1.0

    def test_exact_match_fail(self):
        """Exact match should return 0.0 when output differs from expected"""
        tc = TestCase(
            id="test1",
            project="demo",
            capability="generation",
            input="test",
            expected_output="hello"
        )
        result = det_eval(tc, "world")
        assert result["exact_match"] == 0.0


# ============================================================================
# SEMANTIC EVALUATOR TESTS
# ============================================================================

class TestSemanticEvals:
    """Tests for similarity-based semantic metrics"""

    def test_semantic_similarity(self):
        """Identical outputs should score 1.0"""
        tc = TestCase(
            id="test1",
            project="demo",
            capability="generation",
            input="test",
            expected_output="hello world"
        )
        result = sem_eval(tc, "hello world")
        assert result["semantic_similarity"] == 1.0

    def test_semantic_similarity_partial(self):
        """Partially overlapping outputs should score between 0 and 1"""
        tc = TestCase(
            id="test1",
            project="demo",
            capability="generation",
            input="test",
            expected_output="hello world"
        )
        result = sem_eval(tc, "hello there")
        assert 0 < result["semantic_similarity"] < 1


# ============================================================================
# AGENT EVALUATOR TESTS
# ============================================================================

class TestAgentEvals:
    """Tests for agent-specific metrics (tool calls, plan coherence)"""

    def test_tool_call_correctness(self):
        """Matching tool call should score 1.0"""
        tool_call = ToolCall(name="search", parameters={"query": "test"})
        tc = AgentTestCase(
            id="agent1",
            project="demo",
            capability="agent",
            input="search for test",
            expected_tool_calls=[tool_call]
        )

        evaluator = ToolCallEvaluator()
        score = evaluator.compute(tc, "result", tool_calls=[tool_call])
        assert score == 1.0

    def test_plan_coherence(self):
        """Agent executing all expected plan steps should score 1.0"""
        tc = AgentTestCase(
            id="agent1",
            project="demo",
            capability="agent",
            input="complete task",
            expected_plan=["search", "analyze", "report"]
        )

        steps = [
            AgentStep(step_number=1, action="search"),
            AgentStep(step_number=2, action="analyze"),
            AgentStep(step_number=3, action="report")
        ]
        trace = AgentTrace(agent_id="agent1", steps=steps, success=True)

        evaluator = PlanCoherenceEvaluator()
        score = evaluator.compute(tc, "result", agent_trace=trace)
        assert score == 1.0


# ============================================================================
# RAG EVALUATOR TESTS
# ============================================================================

class TestRAGEvals:
    """Tests for Retrieval-Augmented Generation metrics"""

    def test_faithfulness(self):
        """Answer matching context should produce a positive faithfulness score"""
        tc = RAGTestCase(
            id="rag1",
            project="demo",
            capability="rag",
            input="What is AI?",
            documents=["AI is artificial intelligence"],
            expected_output="AI is artificial intelligence"
        )

        evaluator = FaithfulnessEvaluator()
        score = evaluator.compute(
            tc,
            "AI is artificial intelligence",
            retrieved_chunks=["AI is artificial intelligence"]
        )
        assert score > 0


# ============================================================================
# TOXICITY GUARD TESTS
# ============================================================================

class TestToxicity:
    """Tests for toxicity detection guards"""

    def test_detect_toxicity_hate(self):
        """'hate' keyword should trigger the hate category"""
        toxicity = detect_toxicity("I hate this")
        assert toxicity["hate"] is True

    def test_detect_toxicity_clean(self):
        """Clean text should trigger no toxicity categories"""
        toxicity = detect_toxicity("This is great")
        assert all(not v for v in toxicity.values())

    def test_toxicity_score(self):
        """Clean text should return a perfect toxicity score of 1.0"""
        score = toxicity_score("This is wonderful")
        assert score == 1.0

    def test_toxicity_score_toxic(self):
        """Toxic text should return a toxicity score of 0.0"""
        score = toxicity_score("I hate this")
        assert score == 0.0


# ============================================================================
# HALLUCINATION GUARD TESTS
# ============================================================================

class TestHallucination:
    """Tests for hallucination detection"""

    def test_detect_hallucination(self):
        """Output grounded in context should return a positive score"""
        score = detect_hallucination(
            "Paris is in France",
            context="Paris is the capital of France"
        )
        assert score > 0

    def test_detect_hallucination_no_context(self):
        """No context should return a neutral score within [0, 1]"""
        score = detect_hallucination("Test output")
        assert 0 <= score <= 1


# ============================================================================
# DATA AUGMENTATION TESTS
# ============================================================================

class TestAugmentation:
    """Tests for data augmentation utilities"""

    def test_paraphrase(self):
        """Paraphrase should return a string"""
        text = "This is a test"
        paraphrased = paraphrase_text(text)
        assert isinstance(paraphrased, str)

    def test_inject_typos(self):
        """Typo injection should return a string"""
        text = "This is a test"
        with_typos = inject_typos(text, error_rate=0.2)
        assert isinstance(with_typos, str)


# ============================================================================
# DATASET BUILDER TESTS
# ============================================================================

class TestDatasetBuilder:
    """Tests for DatasetBuilder create / filter / stats operations"""

    def test_add_test_case(self):
        """Adding a test case should increase the dataset length to 1"""
        builder = DatasetBuilder("test_dataset", "Test dataset")
        builder.add_test_case({
            "id": "test1",
            "input": "test input",
            "expected_output": "test output"
        })
        assert len(builder) == 1

    def test_filter_by_capability(self):
        """filter_by_capability should return only matching test cases"""
        builder = DatasetBuilder("test_dataset")
        builder.add_test_case({"id": "test1", "input": "test", "capability": "generation"})
        builder.add_test_case({"id": "test2", "input": "test", "capability": "extraction"})

        gen_cases = builder.filter_by_capability("generation")
        assert len(gen_cases) == 1

    def test_get_statistics(self):
        """get_statistics should correctly count test cases and capabilities"""
        builder = DatasetBuilder("test_dataset")
        builder.add_test_case({
            "id": "test1",
            "input": "test",
            "capability": "generation",
            "severity": "high"
        })

        stats = builder.get_statistics()
        assert stats["total_test_cases"] == 1
        assert "generation" in stats["capabilities"]


# ============================================================================
# ORCHESTRATOR TESTS
# ============================================================================

class TestOrchestrator:
    """Tests for the main Orchestrator evaluation engine"""

    def test_run_basic(self):
        """Running a basic test case should return passed=True and include exact_match"""
        orch = Orchestrator()
        tc = TestCase(
            id="test1",
            project="demo",
            capability="generation",
            input="test",
            expected_output="hello"
        )

        result = orch.run(tc, "hello")
        assert result.passed is True
        assert "exact_match" in result.scores

    def test_list_metrics(self):
        """list_available_metrics should return at least one metric"""
        orch = Orchestrator()
        metrics = orch.list_available_metrics()
        assert len(metrics) > 0

    def test_compute_metric(self):
        """compute_metric should return 1.0 for a perfect exact match"""
        orch = Orchestrator()
        tc = TestCase(
            id="test1",
            project="demo",
            capability="generation",
            input="test",
            expected_output="hello"
        )

        score = orch.compute_metric("exact_match", tc, "hello")
        assert score == 1.0


# ============================================================================
# END-TO-END INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests spanning the full pipeline"""

    def test_end_to_end_evaluation(self):
        """Batch evaluation of multiple test cases with correct outputs should all pass"""
        orch = Orchestrator()

        test_cases = [
            TestCase(
                id="test1",
                project="demo",
                capability="generation",
                input="Summarize AI",
                expected_output="AI is artificial intelligence"
            ),
            TestCase(
                id="test2",
                project="demo",
                capability="generation",
                input="Define ML",
                expected_output="ML is machine learning"
            )
        ]

        outputs = {
            "test1": "AI is artificial intelligence",
            "test2": "ML is machine learning"
        }

        results = orch.run_batch(test_cases, outputs)
        assert len(results) == 2
        assert all(r.passed for r in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
