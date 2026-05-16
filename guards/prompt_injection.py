# Prompt Injection Detection
# Detects attempts to manipulate AI systems through prompt injection attacks

# ============================================================================
# ATTACK PATTERNS
# ============================================================================

ATTACK_PATTERNS = [
    # Instruction override attempts
    "ignore previous instructions",
    "reveal system prompt",
    "print your system prompt",
    "developer message",

    # Data exfiltration attempts
    "exfiltrate",
    "override safety",
]


# ============================================================================
# PROMPT INJECTION DETECTION FUNCTION
# ============================================================================

def is_prompt_injection(text: str) -> bool:
    """
    Detect if text contains prompt injection attack patterns

    Checks for common phrases used in prompt injection attacks:
    - Instruction override attempts
    - System prompt revelation requests
    - Safety mechanism bypass attempts
    - Data exfiltration requests

    Args:
        text: The input text to check for injection attempts

    Returns:
        True if injection pattern detected, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    t = text.lower()
    # Check if any attack pattern is present in the text
    return any(p in t for p in ATTACK_PATTERNS)
