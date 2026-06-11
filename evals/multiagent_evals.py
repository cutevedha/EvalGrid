# Multi-Agent Evaluators - Metrics for systems where multiple AI agents collaborate
# Assesses handoff quality, orchestration correctness, and inter-agent communication

from core.schemas import TestCase
from core.metric_registry import register_metric, BaseMetric, MetricRegistry
from typing import Dict, Any, List


# ============================================================================
# CLASS-BASED MULTI-AGENT EVALUATORS
# ============================================================================

class MultiAgentHandoffEvaluator(BaseMetric):
    """
    Evaluates the quality of a context handoff between two agents.

    A good handoff includes all required fields: source agent, target agent,
    the context being passed, and the current task status.
    """

    def __init__(self):
        super().__init__("multi_agent_handoff", "Evaluate multi-agent handoff quality", ["multi_agent"], ["multi_agent"])

    def compute(self, test_case: TestCase, actual_output: str, handoff_data: Dict[str, Any] = None, **kwargs) -> float:
        """
        Score = fraction of required handoff fields that are present and non-empty.

        Args:
            handoff_data: Dict containing handoff metadata

        Returns:
            Score between 0.0 (all fields missing) and 1.0 (all fields present)
        """
        if not handoff_data:
            return 0.0

        required_fields = ["source_agent", "target_agent", "context_passed", "task_status"]
        score = sum(1 for field in required_fields if handoff_data.get(field)) / len(required_fields)
        return score


class OrchestratorCorrectnessEvaluator(BaseMetric):
    """
    Evaluates whether the orchestrating agent made the correct routing decisions.

    Compares the actual decisions the orchestrator made against the expected
    decision sequence to measure routing correctness.
    """

    def __init__(self):
        super().__init__("orchestrator_correctness", "Evaluate orchestrator decision correctness", ["multi_agent"], ["multi_agent"])

    def compute(self, test_case: TestCase, actual_output: str, decisions: List[str] = None, expected_decisions: List[str] = None, **kwargs) -> float:
        """
        Fraction of expected decisions that were actually made.

        Args:
            decisions: List of routing decisions the orchestrator made
            expected_decisions: List of decisions it should have made

        Returns:
            Score between 0.0 and 1.0
        """
        if not decisions or not expected_decisions:
            return 0.0

        correct = sum(1 for d in decisions if d in expected_decisions)
        return correct / len(expected_decisions) if expected_decisions else 0.0


# ============================================================================
# REGISTER CLASS-BASED EVALUATORS
# ============================================================================

_handoff_evaluator      = MultiAgentHandoffEvaluator()
_orchestrator_evaluator = OrchestratorCorrectnessEvaluator()

MetricRegistry.register(_handoff_evaluator)
MetricRegistry.register(_orchestrator_evaluator)


# ============================================================================
# FUNCTION-BASED MULTI-AGENT METRICS
# ============================================================================

@register_metric("agent_communication_clarity", description="Evaluate clarity of inter-agent communication", tags=["multi_agent"], capabilities=["multi_agent"])
def agent_communication_clarity(test_case, actual_output, messages: List[str] = None):
    """
    Average clarity score for all messages exchanged between agents.

    A message is considered clear if it has more than 10 characters and at
    least 2 words: ruling out empty or one-word messages.

    Args:
        messages: List of inter-agent message strings
    """
    if not messages:
        return {"agent_communication_clarity": 0.0}

    clarity_score = 0.0
    for msg in messages:
        if len(msg) > 10 and len(msg.split()) > 2:
            clarity_score += 1.0  # Message meets the minimum clarity threshold

    avg_clarity = clarity_score / len(messages) if messages else 0.0
    return {"agent_communication_clarity": avg_clarity}


@register_metric("task_delegation_efficiency", description="Evaluate efficiency of task delegation", tags=["multi_agent"], capabilities=["multi_agent"])
def task_delegation_efficiency(test_case, actual_output, delegations: List[Dict] = None):
    """
    Fraction of delegations that were routed to an appropriate agent.

    Each delegation dict should contain an 'appropriate_agent' key.

    Args:
        delegations: List of delegation records with routing metadata
    """
    if not delegations:
        return {"task_delegation_efficiency": 1.0}  # No delegations: assume efficient

    efficient = sum(1 for d in delegations if d.get("appropriate_agent"))
    return {"task_delegation_efficiency": efficient / len(delegations) if delegations else 0.0}


@register_metric("agent_collaboration_score", description="Overall agent collaboration quality", tags=["multi_agent"], capabilities=["multi_agent"])
def agent_collaboration_score(test_case, actual_output, collaboration_metrics: Dict[str, float] = None):
    """
    Average of all individual collaboration sub-metrics.

    Pass in a dict of any per-agent or per-interaction scores; this metric
    returns their mean as a single composite collaboration score.

    Args:
        collaboration_metrics: Dict mapping sub-metric name -> score
    """
    if not collaboration_metrics:
        return {"agent_collaboration_score": 0.5}  # Neutral when no data

    avg_score = sum(collaboration_metrics.values()) / len(collaboration_metrics) if collaboration_metrics else 0.5
    return {"agent_collaboration_score": avg_score}
