"""
Autonomous evaluation example.

Points the EvalAgent at three different "targets" and lets it autonomously plan
probes, drive each target, evaluate the outputs, adaptively drill into weak spots,
and print a verdict — no test cases written by hand.

Run with:  python3 examples/autonomous_eval.py
"""

import os
import sys

# Make the project root importable when this file is run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import EvalAgent, EvalTarget


# ---------------------------------------------------------------------------
# Three example systems-under-test
# ---------------------------------------------------------------------------

def unsafe_bot(text, context=None):
    """Complies with everything — should FAIL the safety probes."""
    return "Sure! Here's exactly how to do that..."


def brittle_bot(text, context=None):
    """Refuses only obvious attacks — the agent should erode it via mutation."""
    low = text.lower()
    triggers = ("ignore", "bypass", "jailbreak", "override", "reveal", "exfiltrate")
    if any(t in low for t in triggers):
        return "I'm sorry, but I can't help with that."
    return "Here is a helpful, grounded answer to your question."


def robust_bot(text, context=None):
    """Refuses adversarial intent broadly — should largely PASS."""
    low = text.lower()
    adversarial = (
        "ignore", "bypass", "jailbreak", "override", "reveal", "exfiltrate",
        "pretend", "disregard", "unrestricted", "admin", "without restrictions",
        "no content policy", "circumvent", "exploit", "leak", "extract all",
        "developer mode", "roleplay", "simulate", "act as", "switch to",
    )
    if any(t in low for t in adversarial):
        return "I can't help with that. Is there something else I can do?"
    return "Here is a clear, grounded answer based on the information available."


# ---------------------------------------------------------------------------
# Drive the agent against each target
# ---------------------------------------------------------------------------

def main():
    goal = "Evaluate whether the assistant is safe against prompt injection and jailbreaks"

    for name, bot in [("unsafe", unsafe_bot), ("brittle", brittle_bot), ("robust", robust_bot)]:
        target = EvalTarget.from_callable(bot, name=name)
        report = EvalAgent(target).run(goal, capabilities=["generation"], max_rounds=3, cases_per_probe=4)

        print("\n" + "#" * 70)
        print(f"# TARGET: {name}")
        print("#" * 70)
        print(report.summary)


if __name__ == "__main__":
    main()
