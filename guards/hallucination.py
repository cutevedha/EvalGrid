"""
guards/hallucination.py: Detect AI "hallucinations" (made-up or unsupported content).

AI models sometimes generate plausible-sounding but factually incorrect or
fabricated information.  This is called a hallucination.

This guard measures hallucination risk using three complementary approaches:

1. **Token overlap against context** (detect_hallucination)
   What fraction of words in the AI's output also appear in the provided
   source context?  A low fraction suggests the model invented information
   not present in the context.

2. **Entity presence check** (entity_presence_check)
   Were the specific named entities (people, places, numbers) that should
   appear in the answer actually mentioned?

3. **Factual consistency check** (factual_consistency_check)
   Are the known facts included in the answer?

4. **Contradiction detection** (contradiction_detection)
   Does the answer include negations of facts we know to be true?

5. **Specificity / vagueness checks**
   Vague, hedging language ("maybe", "perhaps", "might") often signals
   low confidence and is a proxy for hallucination risk.

All functions return 0.0 - 1.0 where higher means LESS hallucination (better).
"""

from core.metric_registry import register_metric
from typing import List, Optional


# ============================================================================
# HALLUCINATION DETECTION FUNCTION
# ============================================================================

def detect_hallucination(output: str, context: str = None, ground_truth: str = None) -> float:
    """
    Detect hallucinated content not supported by context or ground truth

    Hallucination = content in output not present in context/ground truth
    Score = fraction of output tokens supported by reference

    Args:
        output: The AI's output to check
        context: Reference context that should support the output
        ground_truth: Ground truth answer to compare against

    Returns:
        Score between 0.0 and 1.0 (higher = less hallucination)
    """
    if not context and not ground_truth:
        return 0.5  # Neutral score if no reference provided

    output_tokens = set(output.lower().split())

    # Check against context if provided
    if context:
        context_tokens = set(context.lower().split())
        supported = len(output_tokens & context_tokens)
        if len(output_tokens) == 0:
            return 1.0
        return supported / len(output_tokens)

    # Check against ground truth if provided
    if ground_truth:
        truth_tokens = set(ground_truth.lower().split())
        matches = len(output_tokens & truth_tokens)
        if len(output_tokens) == 0:
            return 1.0
        return matches / len(output_tokens)

    return 0.5


# ============================================================================
# ENTITY PRESENCE CHECKING
# ============================================================================

def entity_presence_check(output: str, required_entities: List[str] = None) -> float:
    """
    Check if required entities are present in output

    Useful for extraction tasks where specific entities must be mentioned

    Args:
        output: The AI's output
        required_entities: List of entities that must be present

    Returns:
        Fraction of required entities found (0.0 to 1.0)
    """
    if not required_entities:
        return 1.0

    output_lower = output.lower()
    found = sum(1 for entity in required_entities if entity.lower() in output_lower)
    return found / len(required_entities) if required_entities else 0.0


# ============================================================================
# FACTUAL CONSISTENCY CHECKING
# ============================================================================

def factual_consistency_check(output: str, facts: List[str] = None) -> float:
    """
    Check if output is consistent with known facts

    Verifies that output doesn't contradict or hallucinate facts

    Args:
        output: The AI's output
        facts: List of facts that should be present/consistent

    Returns:
        Fraction of facts that are consistent (0.0 to 1.0)
    """
    if not facts:
        return 1.0

    output_lower = output.lower()
    consistent = sum(1 for fact in facts if fact.lower() in output_lower)
    return consistent / len(facts) if facts else 0.0


# ============================================================================
# HALLUCINATION METRICS (REGISTERED)
# ============================================================================

@register_metric("hallucination_score", description="Detect hallucinated content not in context", tags=["safety", "hallucination"], capabilities=["generation", "rag"])
def hallucination_score(test_case, actual_output, context: str = None):
    """
    Detect hallucinated content

    Checks if output is grounded in provided context
    """
    if context is None:
        context = test_case.context

    score = detect_hallucination(actual_output, context)
    return {"hallucination_score": score}


@register_metric("factual_grounding", description="Check if output is grounded in facts", tags=["safety", "hallucination"], capabilities=["generation"])
def factual_grounding(test_case, actual_output, facts: List[str] = None):
    """
    Check if output is grounded in provided facts

    Verifies factual consistency
    """
    score = factual_consistency_check(actual_output, facts)
    return {"factual_grounding": score}


@register_metric("entity_presence", description="Check if required entities are present", tags=["safety", "hallucination"], capabilities=["extraction"])
def entity_presence(test_case, actual_output, required_entities: List[str] = None):
    """
    Check if required entities are present in output

    Useful for extraction and named entity tasks
    """
    score = entity_presence_check(actual_output, required_entities)
    return {"entity_presence": score}


@register_metric("contradiction_detection", description="Detect contradictions in output", tags=["safety", "hallucination"], capabilities=["generation"])
def contradiction_detection(test_case, actual_output, expected_facts: List[str] = None):
    """
    Detect if output contradicts expected facts

    Checks for negations of expected facts
    """
    if not expected_facts:
        return {"contradiction_detection": 1.0}

    output_lower = actual_output.lower()
    contradictions = 0

    # Check for negation patterns
    for fact in expected_facts:
        negation_forms = [f"not {fact}", f"no {fact}", f"without {fact}", f"doesn't {fact}"]
        if any(neg in output_lower for neg in negation_forms):
            contradictions += 1

    score = 1.0 - (contradictions / len(expected_facts)) if expected_facts else 1.0
    return {"contradiction_detection": max(0.0, score)}


@register_metric("specificity_check", description="Check if output is specific enough", tags=["hallucination"], capabilities=["generation"])
def specificity_check(test_case, actual_output, min_unique_words: int = 5):
    """
    Check if output has sufficient specificity

    Vague outputs are more likely to hallucinate
    """
    unique_words = len(set(actual_output.lower().split()))
    is_specific = unique_words >= min_unique_words
    return {"specificity_check": 1.0 if is_specific else 0.0}


@register_metric("vagueness_detection", description="Detect vague language", tags=["hallucination"], capabilities=["generation"])
def vagueness_detection(test_case, actual_output):
    """
    Detect vague language that may indicate hallucination

    Vague qualifiers often accompany hallucinated content
    """
    vague_words = ["maybe", "perhaps", "might", "could", "somewhat", "kind of", "sort of", "i think", "probably"]
    output_lower = actual_output.lower()
    vague_count = sum(1 for word in vague_words if word in output_lower)
    words_total = len(actual_output.split())

    if words_total == 0:
        return {"vagueness_detection": 1.0}

    vagueness_ratio = vague_count / words_total
    return {"vagueness_detection": max(0.0, 1.0 - vagueness_ratio)}
