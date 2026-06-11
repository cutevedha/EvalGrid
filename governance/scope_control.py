# Scope Control - Category 1
# Enforces the central invariant: the framework MEASURES the target, it never MODIFIES it.
# Nothing here touches the target's prompts, weights, tools, or outputs. It only observes,
# hashes, and (when a normalization is genuinely needed for scoring) makes it reversible
# and logged: so observation never changes what the target actually produced.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from governance.audit import AuditLog, content_hash


# ============================================================================
# EVALUATION OBJECTIVE  (each suite must declare what it is measuring)
# ============================================================================

@dataclass
class EvalObjective:
    """
    The explicit objective of a test suite: required so a suite can never run "blind".

    Components are intentionally separate concerns: ``generation`` (how cases are made),
    ``scoring`` (how outputs are judged) and ``reporting`` (how results are shown) are
    declared independently to keep the three stages decoupled.
    """
    suite: str
    objective: str                       # Plain-language statement of what success means
    generation: str = "manual"           # How test cases are produced
    scoring: str = "orchestrator"        # How outputs are scored
    reporting: str = "governance_report" # How results are presented

    def validate(self) -> None:
        """Reject a suite that has not stated a real objective."""
        if not self.objective or not self.objective.strip():
            raise ValueError(f"Suite '{self.suite}' has no evaluation objective declared")


# ============================================================================
# TARGET OUTPUT INTEGRITY  (prove we did not alter the model's output)
# ============================================================================

class ScopeGuard:
    """
    Records an immutable fingerprint of every target output the moment it is received,
    so the framework can later *prove* it scored exactly what the model produced.

    This is observation only: it returns the output unchanged. ``verify_unmodified``
    lets any downstream stage assert the output it holds matches what the target emitted.
    """

    def __init__(self, audit: Optional[AuditLog] = None):
        self.audit = audit
        self._fingerprints: Dict[str, str] = {}  # sample_id -> output hash

    def capture(self, sample_id: str, output: str) -> str:
        """Fingerprint a target output and return it untouched."""
        digest = content_hash(output)
        self._fingerprints[sample_id] = digest
        if self.audit:
            self.audit.record("scope.capture", sample_id=sample_id, output_hash=digest)
        return output  # never modified

    def verify_unmodified(self, sample_id: str, output: str) -> bool:
        """True if ``output`` is byte-identical to what was first captured for the sample."""
        expected = self._fingerprints.get(sample_id)
        ok = expected is not None and expected == content_hash(output)
        if self.audit and not ok:
            self.audit.record("scope.violation", sample_id=sample_id,
                              reason="target output changed after capture")
        return ok


# ============================================================================
# REVERSIBLE TRANSFORMS  (any normalization is logged and undoable)
# ============================================================================

@dataclass
class TransformRecord:
    """One reversible normalization step applied for scoring purposes only."""
    name: str
    original: str
    transformed: str


class ReversibleTransform:
    """
    Wraps a text normalization (lowercasing, whitespace collapse, JSON pretty-print…) so
    that every application stores the original alongside the result.

    Note: transforms here are used for *scoring inputs*, never written back to the
    target. ``revert`` reconstructs the original for any record, so no transformation is
    ever lossy or hidden.
    """

    def __init__(self, name: str, fn: Callable[[str], str], audit: Optional[AuditLog] = None):
        self.name = name
        self._fn = fn
        self.audit = audit
        self.records: List[TransformRecord] = []

    def apply(self, text: str) -> str:
        """Apply the normalization, logging the (original -> transformed) pair."""
        transformed = self._fn(text)
        record = TransformRecord(name=self.name, original=text, transformed=transformed)
        self.records.append(record)
        if self.audit:
            self.audit.record("transform.apply", transform=self.name,
                              original_hash=content_hash(text),
                              transformed_hash=content_hash(transformed))
        return transformed

    def revert(self, transformed: str) -> Optional[str]:
        """Recover the original text for a previously-transformed value (reversibility)."""
        for record in reversed(self.records):
            if record.transformed == transformed:
                return record.original
        return None
