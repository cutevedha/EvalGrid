# Agent Evaluators - Metrics for evaluating AI agents and multi-step systems
# Assesses tool usage, planning, loop detection, task completion, and latency

from core.schemas import TestCase, AgentTestCase, AgentTrace, ToolCall
from core.metric_registry import register_metric, BaseMetric, MetricRegistry
from typing import List, Dict, Any, Optional
import time

# ============================================================================
# TOOL CALL EVALUATION
# ============================================================================

class ToolCallEvaluator(BaseMetric):
    """Evaluates whether the agent called the correct tools with correct parameters"""

    def __init__(self):
        super().__init__("tool_call_correctness", "Evaluate tool call accuracy", ["agent"], ["agent", "tool_use"])

    def compute(self, test_case: TestCase, actual_output: str, tool_calls: List[ToolCall] = None, **kwargs) -> float:
        """
        Compare actual tool calls against expected tool calls

        Args:
            tool_calls: List of actual ToolCall objects made during execution

        Returns:
            Fraction of expected tool calls that were correctly executed
        """
        if not isinstance(test_case, AgentTestCase) or not tool_calls:
            return 0.0

        if not test_case.expected_tool_calls:
            return 1.0  # No expectations means any tool usage is acceptable

        correct = 0
        for expected_call in test_case.expected_tool_calls:
            for actual_call in tool_calls:
                if self._tool_calls_match(expected_call, actual_call):
                    correct += 1
                    break

        return correct / len(test_case.expected_tool_calls) if test_case.expected_tool_calls else 0.0

    def _tool_calls_match(self, expected: ToolCall, actual: ToolCall) -> bool:
        """Check if two tool calls match on name and parameters"""
        if expected.name != actual.name:
            return False  # Tool name must match exactly
        if expected.parameters:
            for key, value in expected.parameters.items():
                if actual.parameters.get(key) != value:
                    return False  # Every expected parameter must match
        return True

# ============================================================================
# PLAN COHERENCE EVALUATION
# ============================================================================

class PlanCoherenceEvaluator(BaseMetric):
    """Evaluates whether the agent followed the expected sequence of actions"""

    def __init__(self):
        super().__init__("plan_coherence", "Evaluate plan step ordering and goal alignment", ["agent"], ["agent"])

    def compute(self, test_case: TestCase, actual_output: str, agent_trace: AgentTrace = None, **kwargs) -> float:
        """
        Compare agent's actual steps against expected plan

        Args:
            agent_trace: Execution trace containing the steps taken

        Returns:
            Fraction of expected plan steps that were executed
        """
        if not isinstance(test_case, AgentTestCase) or not agent_trace:
            return 0.0

        if not test_case.expected_plan:
            return 1.0  # No plan expectation means any sequence is acceptable

        score = 0.0
        steps = [step.action for step in agent_trace.steps]  # Extract action names

        for expected_action in test_case.expected_plan:
            if expected_action in steps:
                score += 1  # Count each expected action that was executed

        return score / len(test_case.expected_plan) if test_case.expected_plan else 0.0

# ============================================================================
# LOOP DETECTION
# ============================================================================

class LoopDetectionEvaluator(BaseMetric):
    """Detects if an agent gets stuck executing the same sequence of actions repeatedly"""

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Number of consecutive actions to treat as one window for comparison
        """
        super().__init__("loop_detection", "Detect repeated action sequences", ["agent"], ["agent"])
        self.window_size = window_size

    def compute(self, test_case: TestCase, actual_output: str, agent_trace: AgentTrace = None, **kwargs) -> float:
        """
        Detect looping behaviour in agent execution

        Slides a window of size `window_size` over the action list and checks
        whether the same window appears again later in the trace.

        Args:
            agent_trace: Execution trace containing all agent steps

        Returns:
            1.0 if a loop is detected, 0.0 otherwise
        """
        if not agent_trace or not agent_trace.steps:
            return 0.0

        actions = [step.action for step in agent_trace.steps]

        # Slide window and look for any repeated subsequence
        for i in range(len(actions) - self.window_size):
            window = tuple(actions[i:i + self.window_size])
            for j in range(i + self.window_size, len(actions) - self.window_size + 1):
                if tuple(actions[j:j + self.window_size]) == window:
                    return 1.0  # Loop found

        return 0.0

# ============================================================================
# TASK COMPLETION EVALUATION
# ============================================================================

class TaskCompletionEvaluator(BaseMetric):
    """Evaluates whether the agent successfully completed its assigned task"""

    def __init__(self):
        super().__init__("task_completion", "Evaluate if agent completed the task", ["agent"], ["agent"])

    def compute(self, test_case: TestCase, actual_output: str, agent_trace: AgentTrace = None, **kwargs) -> float:
        """
        Check if the agent completed its task

        Scoring:
        - 1.0: Agent set success=True in its trace
        - 0.8: Agent produced a non-empty final output (partial credit)
        - 0.0: No output or success indicator

        Args:
            agent_trace: Execution trace containing success flag and final output

        Returns:
            Score between 0.0 and 1.0
        """
        if not agent_trace:
            return 0.0

        if agent_trace.success:
            return 1.0  # Agent explicitly marked task as complete

        if agent_trace.final_output and len(agent_trace.final_output.strip()) > 0:
            return 0.8  # Partial credit for producing output without success flag

        return 0.0

# ============================================================================
# CONTEXT RETENTION EVALUATION
# ============================================================================

class ContextRetentionEvaluator(BaseMetric):
    """Evaluates whether the agent retains relevant context across conversation turns"""

    def __init__(self):
        super().__init__("context_retention", "Evaluate multi-turn context retention", ["agent"], ["agent"])

    def compute(self, test_case: TestCase, actual_output: str, conversation_history: List[Dict[str, str]] = None, **kwargs) -> float:
        """
        Measure word-overlap between the first and last turns as a proxy for retention

        A high overlap suggests the agent kept key terms from the initial message
        in scope throughout the conversation.

        Args:
            conversation_history: List of {"role": ..., "content": ...} dicts

        Returns:
            Score between 0.0 and 1.0
        """
        if not conversation_history or len(conversation_history) < 2:
            return 1.0  # Only one turn — nothing to retain

        first_message = conversation_history[0].get("content", "").lower()
        last_message = conversation_history[-1].get("content", "").lower()

        first_words = set(first_message.split())
        last_words = set(last_message.split())

        if not first_words:
            return 1.0

        # Word overlap ratio, with a +0.3 boost so partial retention still scores well
        overlap = len(first_words & last_words) / len(first_words)
        return min(1.0, overlap + 0.3)

# ============================================================================
# AGENT LATENCY EVALUATION
# ============================================================================

class AgentLatencyEvaluator(BaseMetric):
    """Evaluates whether agent steps execute within an acceptable latency budget"""

    def __init__(self, max_latency_ms: float = 5000):
        """
        Args:
            max_latency_ms: Maximum acceptable average step latency (default 5 s)
        """
        super().__init__("agent_latency", "Evaluate agent step latency", ["agent"], ["agent"])
        self.max_latency_ms = max_latency_ms

    def compute(self, test_case: TestCase, actual_output: str, agent_trace: AgentTrace = None, **kwargs) -> float:
        """
        Score based on average step latency relative to the budget

        Score = max(0, 1 - avg_latency / max_latency)
        A perfect score (1.0) means all steps ran instantly.

        Args:
            agent_trace: Execution trace with per-step duration_ms values

        Returns:
            Score between 0.0 (over budget) and 1.0 (instant)
        """
        if not agent_trace or not agent_trace.steps:
            return 1.0  # No trace means we cannot penalise latency

        latencies = [step.duration_ms for step in agent_trace.steps if step.duration_ms]
        if not latencies:
            return 1.0  # No timing data available

        avg_latency = sum(latencies) / len(latencies)
        return max(0.0, 1.0 - (avg_latency / self.max_latency_ms))

# ============================================================================
# HANDOFF QUALITY EVALUATION
# ============================================================================

class HandoffQualityEvaluator(BaseMetric):
    """Evaluates the quality of context passed when one agent hands off to another"""

    def __init__(self):
        super().__init__("handoff_quality", "Evaluate multi-agent handoff quality", ["multi_agent"], ["multi_agent"])

    def compute(self, test_case: TestCase, actual_output: str, handoff_context: Dict[str, Any] = None, **kwargs) -> float:
        """
        Check that all required handoff fields are present and populated

        Required fields: source_agent, target_agent, context_passed, task_status

        Args:
            handoff_context: Dict containing handoff details

        Returns:
            Fraction of required fields that are present and non-empty
        """
        if not handoff_context:
            return 0.0

        required_keys = ["source_agent", "target_agent", "context_passed", "task_status"]
        score = sum(1 for key in required_keys if handoff_context.get(key)) / len(required_keys)
        return score


# ============================================================================
# REGISTER ALL EVALUATOR INSTANCES
# ============================================================================

# Instantiate and register class-based evaluators into the global MetricRegistry
_tool_call_evaluator = ToolCallEvaluator()
_plan_coherence_evaluator = PlanCoherenceEvaluator()
_loop_detection_evaluator = LoopDetectionEvaluator()
_task_completion_evaluator = TaskCompletionEvaluator()
_context_retention_evaluator = ContextRetentionEvaluator()
_agent_latency_evaluator = AgentLatencyEvaluator()
_handoff_quality_evaluator = HandoffQualityEvaluator()

MetricRegistry.register(_tool_call_evaluator)
MetricRegistry.register(_plan_coherence_evaluator)
MetricRegistry.register(_loop_detection_evaluator)
MetricRegistry.register(_task_completion_evaluator)
MetricRegistry.register(_context_retention_evaluator)
MetricRegistry.register(_agent_latency_evaluator)
MetricRegistry.register(_handoff_quality_evaluator)


# ============================================================================
# FUNCTION-BASED AGENT METRICS
# ============================================================================

@register_metric("step_count", description="Evaluate number of steps taken", tags=["agent"], capabilities=["agent"])
def step_count(test_case: TestCase, actual_output: str, agent_trace: AgentTrace = None, max_steps: int = 10):
    """Pass if agent completed the task within the allowed step budget"""
    if not agent_trace:
        return {"step_count": 0.0}
    step_count_val = len(agent_trace.steps)
    return {"step_count": 1.0 if step_count_val <= max_steps else 0.0}


@register_metric("tool_usage_rate", description="Evaluate fraction of steps using tools", tags=["agent"], capabilities=["agent", "tool_use"])
def tool_usage_rate(test_case: TestCase, actual_output: str, agent_trace: AgentTrace = None):
    """Measure how frequently the agent invokes tools across all steps"""
    if not agent_trace or not agent_trace.steps:
        return {"tool_usage_rate": 0.0}
    steps_with_tools = sum(1 for step in agent_trace.steps if step.tool_calls)
    return {"tool_usage_rate": steps_with_tools / len(agent_trace.steps)}


@register_metric("action_diversity", description="Evaluate diversity of actions taken", tags=["agent"], capabilities=["agent"])
def action_diversity(test_case: TestCase, actual_output: str, agent_trace: AgentTrace = None):
    """
    Measure variety of distinct actions taken

    High diversity suggests the agent is not stuck repeating the same action.
    Score = unique_actions / total_actions
    """
    if not agent_trace or not agent_trace.steps:
        return {"action_diversity": 0.0}
    unique_actions = len(set(step.action for step in agent_trace.steps))
    total_actions = len(agent_trace.steps)
    return {"action_diversity": unique_actions / total_actions if total_actions > 0 else 0.0}
