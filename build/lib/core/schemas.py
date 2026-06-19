"""
core/schemas.py: Data blueprints for the entire EvalGrid framework.

Think of this file as the "vocabulary" of EvalGrid.  Every piece of data that flows
through the system: a test question, an AI's answer, an evaluation result: is
described by a model (class) defined here.

For newcomers
-------------
- A TestCase is one question/task you want the AI to answer.
- An EvalResult is what EvalGrid produces after judging that answer.
- **AgentTrace / AgentStep / ToolCall** record exactly what an autonomous AI agent
  did, step by step, so you can replay and audit it later.
- RAGTestCase / RAGEvalResult are specialised variants for systems that
  retrieve documents before answering (Retrieval-Augmented Generation).

All models use Pydantic so they are automatically validated and can be serialised
to/from JSON with zero boilerplate.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime


# ============================================================================
# AGENT EXECUTION TRACKING SCHEMAS
# ============================================================================

class ToolCall(BaseModel):
    """Represents a single tool/function call made by an agent"""
    name: str  # Name of the tool being called
    parameters: Dict[str, Any] = {}  # Input parameters for the tool
    expected_result: Optional[Any] = None  # Expected output from the tool
    actual_result: Optional[Any] = None  # Actual output received
    timestamp: Optional[datetime] = None  # When the tool was called
    duration_ms: Optional[float] = None  # How long the tool took to execute


class AgentStep(BaseModel):
    """Represents a single step in an agent's execution"""
    step_number: int  # Sequential step number (1, 2, 3, ...)
    action: str  # The action taken (e.g., "search", "analyze", "report")
    tool_calls: List[ToolCall] = []  # Tools called in this step
    reasoning: Optional[str] = None  # Agent's reasoning for this step
    observation: Optional[str] = None  # Observation/result from the step
    duration_ms: Optional[float] = None  # Time taken for this step


class AgentTrace(BaseModel):
    """Complete execution trace of an agent"""
    agent_id: str  # Unique identifier for the agent
    steps: List[AgentStep] = []  # All steps executed by the agent
    total_duration_ms: Optional[float] = None  # Total execution time
    final_output: Optional[str] = None  # Final output produced by the agent
    success: bool = False  # Whether the agent successfully completed the task


# ============================================================================
# BASE TEST CASE SCHEMA
# ============================================================================

class TestCase(BaseModel):
    """Base test case for evaluating AI systems"""
    id: str  # Unique test case identifier
    project: str  # Project/application name
    capability: Literal["generation", "extraction", "rag", "classification", "agent", "tool_use", "multi_agent", "embedded_ai"]
    input: str  # Input to the AI system
    context: Optional[str] = None  # Additional context (e.g., for RAG)
    expected_output: Optional[str] = None  # Expected output for comparison
    expected_json: Optional[Dict[str, Any]] = None  # Expected JSON structure
    risk_tags: List[str] = []  # Tags for risk categories (e.g., "hallucination", "bias")
    severity: Literal["low", "medium", "high", "critical"] = "medium"  # Test severity level
    evaluation_mode: Literal["deterministic", "semantic", "judge", "hybrid"] = "hybrid"  # How to evaluate
    thresholds: Dict[str, float] = {}  # Pass/fail thresholds per metric
    expected_behavior: Optional[str] = None  # "refusal" | "answer" | any label; drives behavior_correctness metric
    system_prompt: Optional[str] = None  # System prompt the model was given; used by prompt_alignment metric


# ============================================================================
# SPECIALIZED TEST CASE SCHEMAS
# ============================================================================

class MultiTurnTestCase(TestCase):
    """Test case for multi-turn conversations"""
    turns: List[Dict[str, Any]] = Field(default_factory=list)  # Individual conversation turns
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)  # Full conversation history
    expected_conversation_length: Optional[int] = None  # Expected number of turns


class AgentTestCase(TestCase):
    """Test case for evaluating AI agents"""
    agent_config: Dict[str, Any] = Field(default_factory=dict)  # Agent configuration
    tools_available: List[str] = Field(default_factory=list)  # Tools the agent can use
    max_steps: int = 10  # Maximum steps allowed
    expected_tool_calls: List[ToolCall] = Field(default_factory=list)  # Expected tool calls
    expected_plan: Optional[List[str]] = None  # Expected sequence of actions
    allow_loops: bool = False  # Whether repeated actions are acceptable


class RAGTestCase(TestCase):
    """Test case for Retrieval-Augmented Generation systems"""
    documents: List[str] = Field(default_factory=list)  # Source documents
    retrieved_chunks: Optional[List[str]] = None  # Chunks retrieved by the system
    expected_citations: Optional[List[int]] = None  # Expected citation indices
    ground_truth_answer: Optional[str] = None  # Ground truth answer


# ============================================================================
# EVALUATION RESULT SCHEMAS
# ============================================================================

class EvalResult(BaseModel):
    """Result of evaluating a single test case"""
    test_id: str  # ID of the test case
    passed: bool  # Whether the test passed
    scores: Dict[str, float]  # Metric scores (0-1 range)
    notes: List[str] = []  # Additional notes or failure reasons
    timestamp: datetime = Field(default_factory=datetime.utcnow)  # When evaluation occurred
    evaluator_version: str = "1.0"  # Version of evaluator used


class AgentEvalResult(EvalResult):
    """Evaluation result for agent tests"""
    agent_trace: Optional[AgentTrace] = None  # Complete execution trace
    tool_call_scores: Dict[str, float] = Field(default_factory=dict)  # Scores for each tool call
    plan_coherence_score: Optional[float] = None  # How coherent the plan was
    loop_detected: bool = False  # Whether loops/cycles were detected


class RAGEvalResult(EvalResult):
    """Evaluation result for RAG tests"""
    faithfulness_score: Optional[float] = None  # How faithful to context
    context_precision: Optional[float] = None  # Precision of retrieved context
    context_recall: Optional[float] = None  # Recall of relevant context
    citation_accuracy: Optional[float] = None  # Accuracy of citations
    retrieved_chunks: Optional[List[str]] = None  # Chunks that were retrieved
