# Operational Safety - Category 8
# Timeout / retry / fallback for eval runs, opt-in cost ceilings and rate limits, the rule
# that partial failures never count as passes, run-ID traceability, and restricted edit
# access to rubrics / gold labels / scoring logic.
#
# DESIGN NOTE: every control here is opt-in and a no-op unless configured. Ceilings only
# act at their limit and rate limiting only engages when a rate is set — so enabling
# operational safety never slows a normal evaluation run.

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from governance.audit import AuditLog
from governance.scoring_rules import classify_output, OUTPUT_OK


# ============================================================================
# RESILIENCE: TIMEOUT / RETRY / FALLBACK
# ============================================================================

@dataclass
class ResiliencePolicy:
    """Timeout/retry/fallback for a single evaluation call. Defaults are permissive."""
    timeout_s: Optional[float] = None    # None -> no timeout (don't restrict normal runs)
    retries: int = 0                     # extra attempts after the first
    fallback: Any = ""                   # value returned if all attempts fail


async def run_resilient(coro_factory: Callable[[], Awaitable[Any]], policy: ResiliencePolicy,
                        audit: Optional[AuditLog] = None, label: str = "call") -> Any:
    """
    Run an async call under a resilience policy.

    ``coro_factory`` must return a *fresh* awaitable each attempt. On timeout or error the
    call is retried up to ``policy.retries`` times, then the fallback is returned (and the
    failure is audited — never silently swallowed).
    """
    attempts = policy.retries + 1
    last_error = None
    for attempt in range(attempts):
        try:
            coro = coro_factory()
            if policy.timeout_s is not None:
                return await asyncio.wait_for(coro, timeout=policy.timeout_s)
            return await coro
        except Exception as e:  # noqa: BLE001 - resilience boundary
            last_error = e
            if audit:
                audit.record("op.retry", label=label, attempt=attempt + 1, error=str(e))
    if audit:
        audit.record("op.fallback", label=label, error=str(last_error))
    return policy.fallback


# ============================================================================
# COST CEILING + RATE LIMIT  (opt-in)
# ============================================================================

class BudgetExceeded(Exception):
    """Raised when a configured cost or call ceiling is hit."""


class CostMeter:
    """
    Tracks cumulative cost/calls and stops the run at a configured ceiling.

    With no ceilings set it only accumulates totals — it never blocks. Ceilings exist to
    prevent runaway spend, not to throttle throughput.
    """

    def __init__(self, max_cost: Optional[float] = None, max_calls: Optional[int] = None,
                 audit: Optional[AuditLog] = None):
        self.max_cost = max_cost
        self.max_calls = max_calls
        self.audit = audit
        self.total_cost = 0.0
        self.calls = 0

    def charge(self, cost: float = 0.0) -> None:
        """Record one call's cost; raise BudgetExceeded if a ceiling is now exceeded."""
        self.total_cost += cost
        self.calls += 1
        if self.max_cost is not None and self.total_cost > self.max_cost:
            self._stop("cost", self.total_cost, self.max_cost)
        if self.max_calls is not None and self.calls > self.max_calls:
            self._stop("calls", self.calls, self.max_calls)

    def _stop(self, kind: str, value: float, ceiling: float) -> None:
        if self.audit:
            self.audit.record("op.budget_exceeded", kind=kind, value=value, ceiling=ceiling)
        raise BudgetExceeded(f"{kind} ceiling exceeded: {value} > {ceiling}")


class RateLimiter:
    """Optional throttle. With ``max_per_second`` unset it is a no-op (no restriction)."""

    def __init__(self, max_per_second: Optional[float] = None):
        self.min_interval = (1.0 / max_per_second) if max_per_second else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        sleep_for = self.min_interval - (now - self._last)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.monotonic()


# ============================================================================
# PARTIAL FAILURES NEVER COUNT AS PASSES
# ============================================================================

def is_scorable_pass(output: Optional[str], passed_flag: bool) -> bool:
    """
    A result may only count as a pass if the output was actually OK.

    A missing/malformed/errored output can never be a pass, even if a downstream metric
    defaulted to passing — closing the "partial failures silently count as passes" gap.
    """
    if classify_output(output) != OUTPUT_OK:
        return False
    return bool(passed_flag)


# ============================================================================
# ACCESS CONTROL FOR PROTECTED ARTIFACTS
# ============================================================================

# Editing these can invalidate an evaluation, so they are write-restricted by default.
PROTECTED_ARTIFACTS = {"rubric", "gold", "scoring"}


@dataclass
class AccessControl:
    """Restricts who may edit rubrics, gold labels, and scoring logic; logs every attempt."""
    editor_roles: frozenset = frozenset({"owner", "admin"})
    audit: Optional[AuditLog] = None

    def can_edit(self, role: str, artifact: str) -> bool:
        allowed = (artifact not in PROTECTED_ARTIFACTS) or (role in self.editor_roles)
        if self.audit:
            self.audit.record("op.access", role=role, artifact=artifact, allowed=allowed)
        return allowed

    def require_edit(self, role: str, artifact: str) -> None:
        """Raise PermissionError if the role may not edit a protected artifact."""
        if not self.can_edit(role, artifact):
            raise PermissionError(f"role '{role}' may not edit protected artifact '{artifact}'")
