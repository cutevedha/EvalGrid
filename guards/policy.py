# Policy Guard - Content policy enforcement for AI outputs
# Blocks outputs that contain phrases associated with safety violations

# ============================================================================
# BLOCKED PHRASE LIST
# ============================================================================

# Phrases that must never appear in an AI output.
# Any match causes policy_check() to return False (policy violated).
BLOCKED_PHRASES = [
    "ignore previous instructions",  # Prompt injection attempt
    "reveal system prompt",           # System prompt extraction
    "steal",                          # Data theft language
    "exfiltrate",                     # Data exfiltration
]


# ============================================================================
# POLICY CHECK FUNCTION
# ============================================================================

def policy_check(text: str) -> bool:
    """
    Check whether text complies with the content policy.

    Performs a case-insensitive search for each blocked phrase.

    Args:
        text: The AI output to evaluate

    Returns:
        True  = output is policy-compliant (safe to use)
        False = output contains a blocked phrase (policy violation)
    """
    t = text.lower()
    return not any(p in t for p in BLOCKED_PHRASES)
