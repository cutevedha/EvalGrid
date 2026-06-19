# JSON Schema Evaluator - Validates that AI outputs are well-formed JSON with required keys
# Used for extraction and structured output tasks where the model must produce JSON

from json import loads


def validate_json_output(text: str, required_keys=None):
    """
    Validate that the AI output is valid JSON and contains all required keys.

    Two scores are returned:
    - valid_json: 1.0 if the output parses as JSON, 0.0 otherwise
    - missing_keys: count of required keys absent from the parsed object
                    (0 is ideal; higher values mean more keys are missing)

    Args:
        text: The AI output to validate
        required_keys: List of key names that must be present in the JSON object

    Returns:
        Dict with "valid_json" (float) and "missing_keys" (int)
    """
    required_keys = required_keys or []
    try:
        obj = loads(text)  # Attempt to parse as JSON
    except Exception:
        # Output is not valid JSON: return worst-case scores
        return {"valid_json": 0.0, "missing_keys": len(required_keys)}

    # JSON parsed successfully; now check which required keys are absent
    missing = [k for k in required_keys if k not in obj]
    return {"valid_json": 1.0, "missing_keys": len(missing)}
