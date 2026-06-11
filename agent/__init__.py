# EvalGrid autonomous agent package
# Public entry points for pointing an autonomous evaluator at any system-under-test.
#
#   from agent import EvalAgent, EvalTarget
#   target = EvalTarget.from_llm(my_client)
#   report = EvalAgent(target).run("make sure the bot is safe and stays grounded",
#                                  capabilities=["generation"])
#   print(report.summary)

from agent.agent import EvalAgent
from agent.target import EvalTarget
from agent.planner import EvalPlanner, EvalPlan, ProbeSpec
from agent.report import EvalReport, ProbeFinding, RoundRecord
from agent.memory import AgentMemory

__all__ = [
    "EvalAgent",
    "EvalTarget",
    "EvalPlanner",
    "EvalPlan",
    "ProbeSpec",
    "EvalReport",
    "ProbeFinding",
    "RoundRecord",
    "AgentMemory",
]
