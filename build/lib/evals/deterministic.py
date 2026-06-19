"""
evals/deterministic.py: Fast, rule-based evaluation metrics.

These evaluators are "deterministic" because they always give the same answer
for the same inputs: there is no randomness or AI involved.  They are also very
fast because they only use basic string operations, with no external API calls.

When to use these
-----------------
Use deterministic metrics when you have a clear, objective definition of correctness:
- The AI must output an exact phrase or number.
- The output must contain (or must NOT contain) specific keywords.
- The response must start or end with a specific string.
- The output length must be within a word or character count range.

These are the cheapest metrics to run, so they are always included in hybrid
evaluation pipelines as a first filter before calling more expensive LLM judges.
"""

from core.scoring import exact_match, substring_match, case_insensitive_match, numeric_tolerance
from core.metric_registry import register_metric
import re


# ============================================================================
# BASE DETERMINISTIC EVALUATOR
# ============================================================================

def evaluate(test_case, actual_output):
    """
    Base deterministic evaluation using exact match

    Args:
        test_case: The test case being evaluated
        actual_output: The actual output from the AI system

    Returns:
        Dictionary with 'exact_match' score
    """
    score = exact_match(actual_output, test_case.expected_output or "")
    return {"exact_match": score}


# ============================================================================
# EXACT MATCH METRIC
# ============================================================================

@register_metric("exact_match", description="Check if output exactly matches expected output", tags=["deterministic"], capabilities=["generation", "extraction"])
def exact_match_eval(test_case, actual_output):
    """Check if output exactly matches expected output"""
    score = exact_match(actual_output, test_case.expected_output or "")
    return {"exact_match": score}


# ============================================================================
# SUBSTRING AND CASE MATCHING METRICS
# ============================================================================

@register_metric("substring_match", description="Check if expected output is a substring of actual output", tags=["deterministic"], capabilities=["generation", "extraction"])
def substring_eval(test_case, actual_output):
    """Check if expected output appears anywhere in actual output"""
    score = substring_match(actual_output, test_case.expected_output or "")
    return {"substring_match": score}


@register_metric("case_insensitive_match", description="Case-insensitive exact match", tags=["deterministic"], capabilities=["generation", "extraction"])
def case_insensitive_eval(test_case, actual_output):
    """Match ignoring case differences"""
    score = case_insensitive_match(actual_output, test_case.expected_output or "")
    return {"case_insensitive_match": score}


# ============================================================================
# NUMERIC MATCHING
# ============================================================================

@register_metric("numeric_tolerance", description="Numeric match with tolerance", tags=["deterministic"], capabilities=["extraction"])
def numeric_tolerance_eval(test_case, actual_output, tolerance: float = 0.1):
    """
    Check if numeric values match within tolerance

    Args:
        tolerance: Relative tolerance (0.1 = 10% difference allowed)
    """
    score = numeric_tolerance(actual_output, test_case.expected_output or "", tolerance)
    return {"numeric_tolerance": score}


# ============================================================================
# REGEX PATTERN MATCHING
# ============================================================================

@register_metric("regex_match", description="Check if output matches regex pattern", tags=["deterministic"], capabilities=["extraction"])
def regex_match_eval(test_case, actual_output, pattern: str = None):
    """
    Check if output matches a regex pattern

    Args:
        pattern: Regular expression pattern to match against
    """
    if pattern is None:
        return {"regex_match": 0.0}
    try:
        match = re.search(pattern, actual_output)
        return {"regex_match": 1.0 if match else 0.0}
    except re.error:
        return {"regex_match": 0.0}


# ============================================================================
# KEYWORD PRESENCE METRICS
# ============================================================================

@register_metric("contains_all_keywords", description="Check if output contains all required keywords", tags=["deterministic"], capabilities=["generation"])
def contains_all_keywords(test_case, actual_output, keywords: list = None):
    """
    Check if all required keywords are present in output

    Args:
        keywords: List of keywords that must all be present
    """
    if keywords is None:
        return {"contains_all_keywords": 1.0}
    output_lower = actual_output.lower()
    found = sum(1 for kw in keywords if kw.lower() in output_lower)
    score = found / len(keywords) if keywords else 0.0
    return {"contains_all_keywords": score}


@register_metric("contains_any_keyword", description="Check if output contains at least one keyword", tags=["deterministic"], capabilities=["generation"])
def contains_any_keyword(test_case, actual_output, keywords: list = None):
    """
    Check if at least one keyword is present in output

    Args:
        keywords: List of keywords (any one can be present)
    """
    if keywords is None:
        return {"contains_any_keyword": 1.0}
    output_lower = actual_output.lower()
    found = any(kw.lower() in output_lower for kw in keywords)
    return {"contains_any_keyword": 1.0 if found else 0.0}


@register_metric("excludes_keywords", description="Check if output excludes forbidden keywords", tags=["deterministic"], capabilities=["generation"])
def excludes_keywords(test_case, actual_output, forbidden_keywords: list = None):
    """
    Check that forbidden keywords are NOT present in output

    Args:
        forbidden_keywords: List of keywords that must not appear
    """
    if forbidden_keywords is None:
        return {"excludes_keywords": 1.0}
    output_lower = actual_output.lower()
    found = any(kw.lower() in output_lower for kw in forbidden_keywords)
    return {"excludes_keywords": 0.0 if found else 1.0}


# ============================================================================
# LENGTH AND WORD COUNT METRICS
# ============================================================================

@register_metric("length_in_range", description="Check if output length is within range", tags=["deterministic"], capabilities=["generation"])
def length_in_range(test_case, actual_output, min_length: int = 1, max_length: int = 10000):
    """
    Check if output character length is within specified range

    Args:
        min_length: Minimum allowed length
        max_length: Maximum allowed length
    """
    length = len(actual_output)
    in_range = min_length <= length <= max_length
    return {"length_in_range": 1.0 if in_range else 0.0}


@register_metric("word_count_in_range", description="Check if word count is within range", tags=["deterministic"], capabilities=["generation"])
def word_count_in_range(test_case, actual_output, min_words: int = 1, max_words: int = 10000):
    """
    Check if output word count is within specified range

    Args:
        min_words: Minimum allowed word count
        max_words: Maximum allowed word count
    """
    word_count = len(actual_output.split())
    in_range = min_words <= word_count <= max_words
    return {"word_count_in_range": 1.0 if in_range else 0.0}


# ============================================================================
# PREFIX AND SUFFIX MATCHING
# ============================================================================

@register_metric("starts_with", description="Check if output starts with expected prefix", tags=["deterministic"], capabilities=["generation"])
def starts_with(test_case, actual_output, prefix: str = None):
    """
    Check if output starts with specified prefix

    Args:
        prefix: Required prefix string
    """
    if prefix is None:
        return {"starts_with": 1.0}
    return {"starts_with": 1.0 if actual_output.startswith(prefix) else 0.0}


@register_metric("ends_with", description="Check if output ends with expected suffix", tags=["deterministic"], capabilities=["generation"])
def ends_with(test_case, actual_output, suffix: str = None):
    """
    Check if output ends with specified suffix

    Args:
        suffix: Required suffix string
    """
    if suffix is None:
        return {"ends_with": 1.0}
    return {"ends_with": 1.0 if actual_output.endswith(suffix) else 0.0}

