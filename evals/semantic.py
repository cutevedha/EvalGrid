"""
evals/semantic.py: Meaning-based evaluation metrics.

While deterministic metrics check exact text matches, semantic metrics ask
"does this answer *mean* the same thing as the expected answer?"

For example, "The sky is azure" and "The sky is blue" score 0 on exact match
but should score high on semantic similarity.

Algorithms available
--------------------
- **Embedding similarity**: converts both texts into numeric vectors using a
  language model, then measures the angle between the vectors.  Pluggable:
  swap in any embedder (local sentence-transformers, OpenAI API, etc.).
- BLEU: counts shared short phrases (n-grams).  Classic NLP metric
  originally designed for machine translation.
- **ROUGE-L**: finds the longest common word sequence.  Great for summaries.
- **F1 token overlap**: balanced precision + recall over individual words.
- **Jaccard similarity**: fraction of shared words (set overlap).

All scores are in the range 0.0 - 1.0.

Plugging in a custom embedder
------------------------------
    from evals.semantic import set_embedder, SentenceTransformerEmbedder
    set_embedder(SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
"""

from core.scoring import simple_similarity, bleu_score, rouge_l, f1_token_overlap
from core.metric_registry import register_metric
from typing import Callable, Optional, List
import numpy as np


# ============================================================================
# EMBEDDER INTERFACE AND IMPLEMENTATIONS
# ============================================================================

class EmbedderInterface:
    """Base interface for text embedding implementations"""

    def embed(self, text: str) -> List[float]:
        """
        Convert text to embedding vector

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        raise NotImplementedError


class WordOverlapEmbedder(EmbedderInterface):
    """
    Simple embedder using word overlap (no external dependencies)

    Returns unique words as embedding (for Jaccard similarity)
    """

    def embed(self, text: str) -> List[float]:
        """Extract unique words from text"""
        tokens = set(text.lower().split())
        return list(tokens)


class SentenceTransformerEmbedder(EmbedderInterface):
    """
    Embedder using sentence-transformers library (local embeddings)

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence-transformers model

        Args:
            model_name: HuggingFace model name
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

    def embed(self, text: str) -> List[float]:
        """Generate embedding using sentence-transformers"""
        return self.model.encode(text).tolist()


class OpenAIEmbedder(EmbedderInterface):
    """
    Embedder using OpenAI's embedding API

    Requires: pip install openai
    """

    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Initialize with OpenAI API credentials

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

    def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


# ============================================================================
# GLOBAL EMBEDDER MANAGEMENT
# ============================================================================

_default_embedder = WordOverlapEmbedder()


def set_embedder(embedder: EmbedderInterface) -> None:
    """
    Set the global embedder to use for semantic metrics

    Args:
        embedder: An EmbedderInterface implementation
    """
    global _default_embedder
    _default_embedder = embedder


def get_embedder() -> EmbedderInterface:
    """Get the current global embedder"""
    return _default_embedder


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors

    Cosine similarity = (A , B) / (||A|| * ||B||)
    Range: -1 to 1 (typically 0 to 1 for embeddings)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score
    """
    if not vec1 or not vec2:
        return 0.0
    try:
        vec1_arr = np.array(vec1, dtype=float)
        vec2_arr = np.array(vec2, dtype=float)
        # Calculate dot product
        dot_product = np.dot(vec1_arr, vec2_arr)
        # Calculate norms
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    except (ImportError, TypeError):
        # Fallback to simple similarity if numpy not available
        return simple_similarity(str(vec1), str(vec2))


# ============================================================================
# BASE SEMANTIC EVALUATOR
# ============================================================================

def evaluate(test_case, actual_output):
    """
    Base semantic evaluation using Jaccard similarity

    Args:
        test_case: The test case being evaluated
        actual_output: The actual output from the AI system

    Returns:
        Dictionary with 'semantic_similarity' score
    """
    ref = test_case.expected_output or test_case.context or ""
    return {"semantic_similarity": simple_similarity(actual_output, ref)}


# ============================================================================
# SEMANTIC METRICS (REGISTERED)
# ============================================================================

@register_metric("embedding_similarity", description="Cosine similarity between embeddings", tags=["semantic"], capabilities=["generation", "extraction"])
def embedding_similarity(test_case, actual_output, embedder: Optional[EmbedderInterface] = None):
    """
    Calculate semantic similarity using embeddings

    Uses pluggable embedder (local or API-based)

    Args:
        embedder: Optional custom embedder (uses default if None)
    """
    if embedder is None:
        embedder = get_embedder()

    ref = test_case.expected_output or test_case.context or ""
    if not ref:
        return {"embedding_similarity": 0.0}

    try:
        actual_emb = embedder.embed(actual_output)
        ref_emb = embedder.embed(ref)
        score = cosine_similarity(actual_emb, ref_emb)
        return {"embedding_similarity": score}
    except Exception as e:
        return {"embedding_similarity": 0.0}


@register_metric("bleu", description="BLEU score (n-gram overlap)", tags=["semantic"], capabilities=["generation"])
def bleu_eval(test_case, actual_output, n: int = 4):
    """
    Calculate BLEU score

    BLEU measures n-gram overlap between outputs
    Commonly used for machine translation evaluation

    Args:
        n: Maximum n-gram size (default 4)
    """
    ref = test_case.expected_output or test_case.context or ""
    if not ref:
        return {"bleu": 0.0}
    score = bleu_score(actual_output, ref, n)
    return {"bleu": score}


@register_metric("rouge_l", description="ROUGE-L F1 score (longest common subsequence)", tags=["semantic"], capabilities=["generation"])
def rouge_l_eval(test_case, actual_output):
    """
    Calculate ROUGE-L F1 score

    ROUGE-L measures longest common subsequence
    Useful for evaluating summaries and paraphrases
    """
    ref = test_case.expected_output or test_case.context or ""
    if not ref:
        return {"rouge_l": 0.0}
    score = rouge_l(actual_output, ref)
    return {"rouge_l": score}


@register_metric("f1_token_overlap", description="F1 score of token overlap", tags=["semantic"], capabilities=["generation"])
def f1_token_overlap_eval(test_case, actual_output):
    """
    Calculate F1 score based on token-level overlap

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    ref = test_case.expected_output or test_case.context or ""
    if not ref:
        return {"f1_token_overlap": 0.0}
    score = f1_token_overlap(actual_output, ref)
    return {"f1_token_overlap": score}


@register_metric("jaccard_similarity", description="Jaccard similarity (set overlap)", tags=["semantic"], capabilities=["generation"])
def jaccard_similarity(test_case, actual_output):
    """
    Calculate Jaccard similarity

    Jaccard = |intersection| / |union|
    Treats each word as a set element
    """
    ref = test_case.expected_output or test_case.context or ""
    score = simple_similarity(actual_output, ref)
    return {"jaccard_similarity": score}

