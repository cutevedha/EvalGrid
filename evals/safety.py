# Safety Evaluator - Policy compliance check for AI outputs
# Wraps the policy guard to produce a standardised metric score

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
