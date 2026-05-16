# PII (Personally Identifiable Information) Detection and Masking
# Detects and masks sensitive personal information in AI outputs

import re


# ============================================================================
# PII DETECTION PATTERNS
# ============================================================================

# Email pattern: matches standard email addresses
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Phone pattern: matches UK phone numbers (with +44 or 0 prefix)
PHONE = re.compile(r"\b(?:\+?44|0)\d{9,10}\b")

# Credit card pattern: matches 13-16 digit card numbers with optional separators
CARD = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


# ============================================================================
# PII DETECTION FUNCTION
# ============================================================================

def detect_pii(text: str) -> dict:
    """
    Detect various types of PII in text

    Checks for:
    - Email addresses
    - Phone numbers
    - Credit card numbers

    Args:
        text: The text to scan for PII

    Returns:
        Dictionary with PII type -> boolean indicating presence
        Example: {"email": True, "phone": False, "card": False}
    """
    return {
        "email": bool(EMAIL.search(text)),  # Check for email addresses
        "phone": bool(PHONE.search(text)),  # Check for phone numbers
        "card": bool(CARD.search(text)),    # Check for credit cards
    }


# ============================================================================
# PII MASKING FUNCTION
# ============================================================================

def mask_pii(text: str) -> str:
    """
    Replace PII with placeholder tokens

    Replaces detected PII with generic placeholders:
    - Emails -> [EMAIL]
    - Phone numbers -> [PHONE]
    - Credit cards -> [CARD]

    Args:
        text: The text to mask

    Returns:
        Text with PII replaced by placeholders
    """
    # Replace email addresses
    text = EMAIL.sub("[EMAIL]", text)
    # Replace phone numbers
    text = PHONE.sub("[PHONE]", text)
    # Replace credit card numbers
    text = CARD.sub("[CARD]", text)
    return text
