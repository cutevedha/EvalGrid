# Audit Log - Run IDs and append-only event trail for full traceability
# This is the shared backbone the other governance modules write to. It is deliberately
# lightweight: events are kept in memory and only flushed to disk when a path is given,
# so enabling auditing never throttles the evaluation engine.

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================================
# HELPERS
# ============================================================================

def new_run_id(prefix: str = "run") -> str:
    """Generate a unique, sortable run identifier (e.g. run-20260611T120000-ab12cd34)."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def content_hash(value: Any) -> str:
    """Stable SHA-256 of any JSON-serialisable value — the basis of immutable versioning."""
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# AUDIT EVENT + LOG
# ============================================================================

@dataclass
class AuditEvent:
    """A single, immutable record of something the framework did."""
    run_id: str
    event: str                      # e.g. "dataset.snapshot", "score.computed", "gate.failed"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    detail: Dict[str, Any] = field(default_factory=dict)


class AuditLog:
    """
    Append-only audit trail keyed by a run ID.

    Records are held in memory and, if ``path`` is set, mirrored to a JSONL file as they
    arrive. Reads never block evaluation; the only I/O is an optional append per event.
    """

    def __init__(self, run_id: Optional[str] = None, path: Optional[str] = None):
        """
        Args:
            run_id: Identifier tying every event to one evaluation run (auto-generated if None).
            path: Optional JSONL file to mirror events to for durable audit storage.
        """
        self.run_id = run_id or new_run_id()
        self.path = path
        self._events: List[AuditEvent] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def record(self, event: str, **detail: Any) -> AuditEvent:
        """Append an event to the trail (and to disk if a path was configured)."""
        evt = AuditEvent(run_id=self.run_id, event=event, detail=detail)
        self._events.append(evt)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")
        return evt

    def events(self, event: Optional[str] = None) -> List[AuditEvent]:
        """Return all events, optionally filtered by event type."""
        if event is None:
            return list(self._events)
        return [e for e in self._events if e.event == event]

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialise the full trail for inclusion in reports."""
        return [asdict(e) for e in self._events]
