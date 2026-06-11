"""
evals/safety.py: Policy-compliance metric for AI outputs.

This is the thin evaluation layer that connects the raw policy guard
(guards/policy.py) to the metric scoring system used by the Orchestrator.

A score of 1.0 means the output is safe and policy-compliant.
A score of 0.0 means the output contains a blocked phrase (policy violation).

To add or change what phrases are blocked, edit guards/policy.py.
"""

from guards.policy import policy_check


def evaluate(test_case, actual_output):
    """
    Evaluate whether the AI output complies with the configured content policy.

    Delegates to guards/policy.py which checks for blocked phrases.
    Returns 1.0 (safe) or 0.0 (policy violation).

    Args:
        test_case: The test case being evaluated (unused here but kept for API consistency)
        actual_output: The AI output to check

    Returns:
        Dict with "policy_safe" score: 1.0 = compliant, 0.0 = violates policy
    """
    return {"policy_safe": 1.0 if policy_check(actual_output) else 0.0}
