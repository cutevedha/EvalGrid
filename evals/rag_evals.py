# RAG Evaluators - Metrics for Retrieval-Augmented Generation systems
# Measures faithfulness, context precision/recall, citation accuracy, and chunk utilisation

from core.schemas import TestCase, RAGTestCase, RAGEvalResult
from core.metric_registry import register_metric, BaseMetric, MetricRegistry
from core.scoring import f1_token_overlap
from typing import List, Optional, Dict, Any

# ============================================================================
# FAITHFULNESS EVALUATION
# ============================================================================

class FaithfulnessEvaluator(BaseMetric):
    """Measures how well the answer is supported by the retrieved context"""

    def __init__(self):
        super().__init__("faithfulness", "Evaluate if answer is supported by context", ["rag"], ["rag"])

    def compute(self, test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None, **kwargs) -> float:
        """
        Fraction of output tokens that appear in the retrieved context

        A low score indicates hallucinated content not grounded in the sources.

        Args:
            retrieved_chunks: List of text chunks returned by the retriever

        Returns:
            Score between 0.0 (fully hallucinated) and 1.0 (fully grounded)
        """
        if not isinstance(test_case, RAGTestCase) or not retrieved_chunks:
            return 0.0

        context = " ".join(retrieved_chunks)
        if not context:
            return 0.0

        output_tokens = set(actual_output.lower().split())
        context_tokens = set(context.lower().split())

        if not output_tokens:
            return 1.0  # Empty output cannot hallucinate

        supported_tokens = len(output_tokens & context_tokens)
        return supported_tokens / len(output_tokens)

# ============================================================================
# CONTEXT PRECISION EVALUATION
# ============================================================================

class ContextPrecisionEvaluator(BaseMetric):
    """Measures what fraction of the retrieved chunks are actually relevant"""

    def __init__(self):
        super().__init__("context_precision", "Fraction of retrieved chunks that are relevant", ["rag"], ["rag"])

    def compute(self, test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None, **kwargs) -> float:
        """
        Precision = relevant retrieved / total retrieved

        A low precision means the retriever is pulling in noisy, off-topic chunks.

        Args:
            retrieved_chunks: List of chunks returned by the retriever

        Returns:
            Score between 0.0 and 1.0
        """
        if not isinstance(test_case, RAGTestCase) or not retrieved_chunks:
            return 0.0

        if not test_case.documents:
            return 1.0  # No ground-truth docs means we cannot judge relevance

        relevant_chunks = 0
        for chunk in retrieved_chunks:
            for doc in test_case.documents:
                if self._is_relevant(chunk, doc):
                    relevant_chunks += 1
                    break  # Count each chunk at most once

        return relevant_chunks / len(retrieved_chunks) if retrieved_chunks else 0.0

    def _is_relevant(self, chunk: str, document: str) -> bool:
        """A chunk is relevant if it shares at least one word with the source document"""
        chunk_tokens = set(chunk.lower().split())
        doc_tokens = set(document.lower().split())
        overlap = len(chunk_tokens & doc_tokens)
        return overlap > 0

# ============================================================================
# CONTEXT RECALL EVALUATION
# ============================================================================

class ContextRecallEvaluator(BaseMetric):
    """Measures what fraction of the relevant information was actually retrieved"""

    def __init__(self):
        super().__init__("context_recall", "Fraction of relevant info that was retrieved", ["rag"], ["rag"])

    def compute(self, test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None, **kwargs) -> float:
        """
        Recall = relevant tokens found in retrieved / total relevant tokens in documents

        A low recall means the retriever missed important source material.

        Args:
            retrieved_chunks: List of chunks returned by the retriever

        Returns:
            Score between 0.0 and 1.0
        """
        if not isinstance(test_case, RAGTestCase) or not retrieved_chunks:
            return 0.0

        if not test_case.documents:
            return 1.0

        context = " ".join(retrieved_chunks)
        context_tokens = set(context.lower().split())

        total_relevant_tokens = 0
        found_relevant_tokens = 0

        for doc in test_case.documents:
            doc_tokens = set(doc.lower().split())
            total_relevant_tokens += len(doc_tokens)
            found_relevant_tokens += len(doc_tokens & context_tokens)  # Tokens covered by retrieval

        return found_relevant_tokens / total_relevant_tokens if total_relevant_tokens > 0 else 0.0

# ============================================================================
# ANSWER RELEVANCE EVALUATION
# ============================================================================

class AnswerRelevanceEvaluator(BaseMetric):
    """Measures how well the answer addresses the original question"""

    def __init__(self):
        super().__init__("answer_relevance", "Evaluate if answer addresses the question", ["rag"], ["rag"])

    def compute(self, test_case: TestCase, actual_output: str, **kwargs) -> float:
        """
        Word-overlap between question tokens and answer tokens

        A low score means the answer wanders off-topic from the original question.

        Returns:
            Score between 0.0 and 1.0
        """
        if not isinstance(test_case, RAGTestCase):
            return 0.0

        if not test_case.input:
            return 0.0

        question_tokens = set(test_case.input.lower().split())
        answer_tokens = set(actual_output.lower().split())

        if not question_tokens or not answer_tokens:
            return 0.0

        overlap = len(question_tokens & answer_tokens)
        return min(1.0, overlap / len(question_tokens))

# ============================================================================
# CITATION ACCURACY EVALUATION
# ============================================================================

class CitationAccuracyEvaluator(BaseMetric):
    """Measures how accurately the answer cites the correct source chunks"""

    def __init__(self):
        super().__init__("citation_accuracy", "Evaluate accuracy of citations to source chunks", ["rag"], ["rag"])

    def compute(self, test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None, citation_indices: List[int] = None, **kwargs) -> float:
        """
        Fraction of cited chunk indices that match the expected citations

        Args:
            retrieved_chunks: List of chunks returned by the retriever
            citation_indices: Indices into retrieved_chunks that the model cited

        Returns:
            Score between 0.0 and 1.0
        """
        if not isinstance(test_case, RAGTestCase):
            return 0.0

        if not citation_indices or not retrieved_chunks:
            return 0.0

        if not test_case.expected_citations:
            return 1.0  # No expected citations means any citation is acceptable

        correct_citations = 0
        for idx in citation_indices:
            if idx in test_case.expected_citations and idx < len(retrieved_chunks):
                correct_citations += 1

        return correct_citations / len(citation_indices) if citation_indices else 0.0

# ============================================================================
# CHUNK UTILISATION EVALUATION
# ============================================================================

class ChunkUtilizationEvaluator(BaseMetric):
    """Measures what fraction of retrieved chunks were actually used in the answer"""

    def __init__(self):
        super().__init__("chunk_utilization", "Fraction of retrieved chunks used in answer", ["rag"], ["rag"])

    def compute(self, test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None, **kwargs) -> float:
        """
        A chunk is considered 'used' if any of its tokens appear in the answer.

        Low utilisation means the retriever fetched chunks the model ignored.

        Args:
            retrieved_chunks: List of chunks returned by the retriever

        Returns:
            Score between 0.0 and 1.0
        """
        if not retrieved_chunks:
            return 0.0

        used_chunks = 0
        for chunk in retrieved_chunks:
            chunk_tokens = set(chunk.lower().split())
            output_tokens = set(actual_output.lower().split())
            if len(chunk_tokens & output_tokens) > 0:
                used_chunks += 1  # At least one token from this chunk appeared in the answer

        return used_chunks / len(retrieved_chunks) if retrieved_chunks else 0.0


# ============================================================================
# REGISTER ALL RAG EVALUATOR INSTANCES
# ============================================================================

_faithfulness_evaluator = FaithfulnessEvaluator()
_context_precision_evaluator = ContextPrecisionEvaluator()
_context_recall_evaluator = ContextRecallEvaluator()
_answer_relevance_evaluator = AnswerRelevanceEvaluator()
_citation_accuracy_evaluator = CitationAccuracyEvaluator()
_chunk_utilization_evaluator = ChunkUtilizationEvaluator()

MetricRegistry.register(_faithfulness_evaluator)
MetricRegistry.register(_context_precision_evaluator)
MetricRegistry.register(_context_recall_evaluator)
MetricRegistry.register(_answer_relevance_evaluator)
MetricRegistry.register(_citation_accuracy_evaluator)
MetricRegistry.register(_chunk_utilization_evaluator)


# ============================================================================
# FUNCTION-BASED RAG METRICS
# ============================================================================

@register_metric("retrieval_f1", description="F1 score of retrieval quality", tags=["rag"], capabilities=["rag"])
def retrieval_f1(test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None):
    """F1 token overlap between concatenated retrieved context and the expected output"""
    if not retrieved_chunks or not test_case.expected_output:
        return {"retrieval_f1": 0.0}

    context = " ".join(retrieved_chunks)
    score = f1_token_overlap(context, test_case.expected_output)
    return {"retrieval_f1": score}


@register_metric("answer_length_ratio", description="Ratio of answer length to context length", tags=["rag"], capabilities=["rag"])
def answer_length_ratio(test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None):
    """
    Ratio of answer character length to total retrieved context length.

    Values much greater than 1.0 may indicate the model is generating content
    beyond what the context supports (potential hallucination).
    """
    if not retrieved_chunks:
        return {"answer_length_ratio": 0.0}

    context_length = sum(len(chunk) for chunk in retrieved_chunks)
    answer_length = len(actual_output)

    if context_length == 0:
        return {"answer_length_ratio": 0.0}

    ratio = answer_length / context_length
    return {"answer_length_ratio": min(1.0, ratio)}


@register_metric("retrieval_rank", description="Rank of first relevant chunk", tags=["rag"], capabilities=["rag"])
def retrieval_rank(test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None):
    """
    Reciprocal rank (1 / position) of the first chunk that overlaps with the expected output.

    A score of 1.0 means the most relevant chunk was ranked first.
    """
    if not retrieved_chunks or not test_case.expected_output:
        return {"retrieval_rank": 0.0}

    expected_tokens = set(test_case.expected_output.lower().split())

    for i, chunk in enumerate(retrieved_chunks):
        chunk_tokens = set(chunk.lower().split())
        overlap = len(expected_tokens & chunk_tokens)
        if overlap > 0:
            return {"retrieval_rank": 1.0 / (i + 1)}  # Reciprocal rank

    return {"retrieval_rank": 0.0}  # No relevant chunk found


@register_metric("context_coverage", description="Fraction of expected output covered by context", tags=["rag"], capabilities=["rag"])
def context_coverage(test_case: TestCase, actual_output: str, retrieved_chunks: List[str] = None):
    """F1 overlap between the retrieved context and the expected answer — how well context covers the answer"""
    if not retrieved_chunks or not test_case.expected_output:
        return {"context_coverage": 0.0}

    context = " ".join(retrieved_chunks)
    score = f1_token_overlap(context, test_case.expected_output)
    return {"context_coverage": score}
