"""
core/scoring.py: Text-similarity and matching algorithms used by the evaluators.

This file is the maths engine of EvalGrid.  When EvalGrid needs to decide "how
close is the AI's answer to the correct answer?", it calls one of the functions
here.

Algorithms available (from cheapest to most complex)
-----------------------------------------------------
1. **Exact / substring / case-insensitive match**: simple string comparisons.
2. **Jaccard similarity**: compares the *set* of words: how many words do both
   texts share?
3. BLEU: counts how many short phrases (n-grams) appear in both texts.
   Widely used to evaluate machine translation.
4. **ROUGE-L**: finds the longest sequence of words that appear in the same
   order in both texts.  Good for summaries.
5. **F1 token overlap**: the harmonic mean of precision and recall at word level.
6. **Edit distance (Levenshtein)**: counts the minimum number of character
   insertions/deletions/replacements needed to turn one text into the other.
7. **Numeric tolerance**: checks whether a numeric answer is within an
   acceptable percentage of the expected value.

All functions return a **float in the range 0.0 - 1.0** where 1.0 means "perfect
match" and 0.0 means "no match at all".
"""

from typing import List
import math


# ============================================================================
# EXACT MATCHING METRICS
# ============================================================================

def exact_match(actual: str, expected: str) -> float:
    """
    Check if actual output exactly matches expected output (case-sensitive)

    Args:
        actual: The actual output from the AI system
        expected: The expected/reference output

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if expected is None:
        return 0.0
    return 1.0 if actual.strip() == expected.strip() else 0.0


def case_insensitive_match(actual: str, expected: str) -> float:
    """
    Check if outputs match ignoring case differences

    Args:
        actual: The actual output
        expected: The expected output

    Returns:
        1.0 if match (case-insensitive), 0.0 otherwise
    """
    return 1.0 if actual.lower().strip() == expected.lower().strip() else 0.0


def substring_match(actual: str, expected: str) -> float:
    """
    Check if expected output appears as substring in actual output

    Args:
        actual: The actual output
        expected: The expected substring

    Returns:
        1.0 if expected is substring of actual, 0.0 otherwise
    """
    return 1.0 if expected.lower() in actual.lower() else 0.0


# ============================================================================
# JACCARD SIMILARITY (SET-BASED)
# ============================================================================

def simple_similarity(actual: str, reference: str) -> float:
    """
    Calculate Jaccard similarity between two texts (word-level overlap).

    Jaccard = |intersection| / |union|
    Treats each unique word as a set element, so word order is ignored.

    Args:
        actual: The AI's output text.
        reference: The expected/reference text.

    Returns:
        Similarity score between 0.0 (no shared words) and 1.0 (identical word sets).
    """
    if not actual or not reference:
        return 0.0
    actual_words = set(actual.lower().split())
    reference_words = set(reference.lower().split())
    shared_words = actual_words & reference_words
    all_words = actual_words | reference_words
    return len(shared_words) / max(len(all_words), 1)


# ============================================================================
# BLEU SCORE (N-GRAM BASED)
# ============================================================================

def bleu_score(actual: str, reference: str, n: int = 4) -> float:
    """
    Calculate BLEU score (Bilingual Evaluation Understudy)

    BLEU measures n-gram overlap between actual and reference
    Commonly used for machine translation evaluation
    Score range: 0.0 to 1.0

    Args:
        actual: The actual output
        reference: The reference/expected output
        n: Maximum n-gram size to consider (default 4)

    Returns:
        BLEU score between 0.0 and 1.0
    """
    actual_tokens = actual.lower().split()
    reference_tokens = reference.lower().split()

    if not actual_tokens or not reference_tokens:
        return 0.0

    precisions = []
    # Calculate precision for each n-gram size
    for i in range(1, min(n + 1, len(actual_tokens) + 1)):
        # Extract n-grams from actual output
        actual_ngrams = set()
        for j in range(len(actual_tokens) - i + 1):
            actual_ngrams.add(tuple(actual_tokens[j:j+i]))

        # Extract n-grams from reference
        reference_ngrams = set()
        for j in range(len(reference_tokens) - i + 1):
            reference_ngrams.add(tuple(reference_tokens[j:j+i]))

        # Calculate precision for this n-gram size
        matches = len(actual_ngrams & reference_ngrams)
        total = len(actual_ngrams)
        precisions.append(matches / total if total > 0 else 0.0)

    # If any n-gram has 0 precision, return 0
    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    # Brevity penalty for short outputs
    brevity_penalty = min(1.0, math.exp(1 - len(reference_tokens) / max(len(actual_tokens), 1)))
    return geo_mean * brevity_penalty


# ============================================================================
# ROUGE-L SCORE (LONGEST COMMON SUBSEQUENCE)
# ============================================================================

def rouge_l(actual: str, reference: str) -> float:
    """
    Calculate ROUGE-L F1 score based on longest common subsequence

    ROUGE-L measures the longest common subsequence between texts
    Useful for evaluating summaries and paraphrases

    Args:
        actual: The actual output
        reference: The reference output

    Returns:
        F1 score between 0.0 and 1.0
    """
    actual_tokens = actual.lower().split()
    reference_tokens = reference.lower().split()

    if not actual_tokens or not reference_tokens:
        return 0.0

    # Find longest common subsequence length
    lcs_length = _lcs_length(actual_tokens, reference_tokens)
    # Calculate recall and precision
    recall = lcs_length / len(reference_tokens)
    precision = lcs_length / len(actual_tokens)

    if recall + precision == 0:
        return 0.0

    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """
    Calculate the length of the longest common subsequence (LCS) using dynamic programming.

    LCS finds the longest sequence of tokens that appears in both inputs in the same
    relative order (but not necessarily contiguous).  Used internally by rouge_l().

    Args:
        seq_a: First token sequence.
        seq_b: Second token sequence.

    Returns:
        Number of tokens in the longest common subsequence.
    """
    rows = len(seq_a)
    cols = len(seq_b)
    # dp[i][j] = LCS length when considering seq_a[0:i] and seq_b[0:j]
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1   # tokens match: extend the LCS
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # take the longer partial LCS

    return dp[rows][cols]


# ============================================================================
# F1 TOKEN OVERLAP
# ============================================================================

def f1_token_overlap(actual: str, expected: str) -> float:
    """
    Calculate F1 score based on token-level overlap

    F1 = 2 * (precision * recall) / (precision + recall)
    Precision = matching tokens / actual tokens
    Recall = matching tokens / expected tokens

    Args:
        actual: The actual output
        expected: The expected output

    Returns:
        F1 score between 0.0 and 1.0
    """
    actual_tokens = set(actual.lower().split())
    expected_tokens = set(expected.lower().split())

    if not actual_tokens or not expected_tokens:
        return 0.0

    # Find matching tokens
    intersection = len(actual_tokens & expected_tokens)
    # Calculate precision and recall
    precision = intersection / len(actual_tokens)
    recall = intersection / len(expected_tokens)

    if precision + recall == 0:
        return 0.0

    # Calculate F1
    return 2 * (precision * recall) / (precision + recall)


# ============================================================================
# EDIT DISTANCE (LEVENSHTEIN DISTANCE)
# ============================================================================

def edit_distance(actual: str, expected: str) -> float:
    """
    Calculate normalized edit distance (Levenshtein distance)

    Edit distance = minimum number of single-character edits needed
    Normalized to 0-1 range where 1.0 = perfect match

    Args:
        actual: The actual output
        expected: The expected output

    Returns:
        Normalized score between 0.0 and 1.0
    """
    actual_lower = actual.lower()
    expected_lower = expected.lower()

    actual_len = len(actual_lower)
    expected_len = len(expected_lower)
    # dp[i][j] = edit distance for actual_lower[0:i] and expected_lower[0:j]
    dp = [[0] * (expected_len + 1) for _ in range(actual_len + 1)]

    # Base cases: converting an empty string requires inserting/deleting every character.
    for i in range(actual_len + 1):
        dp[i][0] = i
    for j in range(expected_len + 1):
        dp[0][j] = j

    for i in range(1, actual_len + 1):
        for j in range(1, expected_len + 1):
            if actual_lower[i - 1] == expected_lower[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]                           # characters match: free
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],                      # delete from actual
                                   dp[i][j - 1],                       # insert from expected
                                   dp[i - 1][j - 1])                   # substitute

    max_len = max(actual_len, expected_len)
    if max_len == 0:
        return 1.0
    # Convert raw distance to a 0-1 similarity: 1.0 = identical, 0.0 = completely different.
    return 1.0 - (dp[actual_len][expected_len] / max_len)


# ============================================================================
# NUMERIC TOLERANCE MATCHING
# ============================================================================

def numeric_tolerance(actual: str, expected: str, tolerance: float = 0.1) -> float:
    """
    Check if numeric values match within a tolerance

    Useful for evaluating numeric outputs (calculations, measurements)
    Tolerance is relative: threshold = expected_value * tolerance

    Args:
        actual: The actual numeric output
        expected: The expected numeric output
        tolerance: Relative tolerance (0.1 = 10%)

    Returns:
        1.0 if within tolerance, 0.0 otherwise
    """
    try:
        actual_val = float(actual.strip())
        expected_val = float(expected.strip())
        # Calculate absolute difference
        diff = abs(actual_val - expected_val)
        # Calculate tolerance threshold
        threshold = abs(expected_val * tolerance)
        return 1.0 if diff <= threshold else 0.0
    except (ValueError, AttributeError):
        return 0.0

