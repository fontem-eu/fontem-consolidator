"""Thin async shim around ``fontem_events.EventLog`` for emitting
events from inside the consolidator (which is otherwise async).

Lazy: nothing is connected unless an event is actually emitted.
If ``EVENTS_DATABASE_URL`` is unset the shim no-ops with a
debug log, so unit tests and dev runs without the event log
still work.

Concurrency: psycopg's sync API blocks the asyncio loop, so each
emit runs through ``asyncio.to_thread``. A single global
``EventLog`` instance is shared; gmr-events' ``EventLog`` keeps
one psycopg connection and ``batch()`` opens a transaction per
call, so concurrent emits serialise on the connection.
"""
from __future__ import annotations

import asyncio
import os
import threading
import uuid

from loguru import logger

_lock = threading.Lock()
# Lazy singleton — naming kept lowercase because it's a mutable cache,
# not a constant. pylint's invalid-name + global-statement complaints
# don't add value for this well-understood double-checked-lock idiom.
_log_singleton = None  # type: ignore[var-annotated]  # pylint: disable=invalid-name


def _get_log():
    """Return the lazily-constructed fontem_events EventLog (or None
    if the env isn't configured)."""
    global _log_singleton  # pylint: disable=global-statement
    if _log_singleton is not None:
        return _log_singleton
    with _lock:
        if _log_singleton is not None:
            return _log_singleton
        if not os.environ.get("EVENTS_DATABASE_URL"):
            logger.debug(
                "eventlog: EVENTS_DATABASE_URL unset, emit will no-op"
            )
            return None
        # Optional dep — imported lazily so unit tests / dev runs without
        # the fontem-events wheel installed don't fail at import time.
        try:
            from fontem_events import EventLog  # pylint: disable=import-outside-toplevel
        except ImportError:  # pragma: no cover
            logger.warning(
                "eventlog: gmr-events not installed, emit will no-op"
            )
            return None
        _log_singleton = EventLog.from_env()
        logger.info("eventlog: connected to events database")
        return _log_singleton


# Five kwargs match the assert_same_as event envelope shape — splitting
# them into a dict would just push the same fields one level deeper.
def _emit_sync(  # pylint: disable=too-many-arguments
    *,
    event_type: str,
    iri: str,
    domain: str,
    payload: dict,
    producer: str,
) -> int | None:
    log = _get_log()
    if log is None:
        return None
    batch_id = uuid.uuid4()
    with log.batch(batch_id, producer=producer) as emit:
        return emit.upsert(
            event_type=event_type,
            iri=iri,
            domain=domain,
            payload=payload,
        )


# Nine kwargs match the AssertSameAs event envelope; this is the public
# surface for emitting "same as" relations from the consolidator and
# packing them into a settings object would just add indirection.
async def emit_assert_same_as(  # pylint: disable=too-many-arguments
    *,
    a_iri: str,
    b_iri: str,
    confidence: float,
    method: str,
    rule: str | None = None,
    tier: str | None = None,
    matched_via_alias: bool = False,
    domain: str = "consolidation",
    producer: str = "fontem-consolidator",
) -> int | None:
    """Emit an AssertSameAs event into ``events.entity_events``.

    Failures are logged but do not propagate — the consolidator's
    Neo4j-side write is the present source of truth; the event
    is what lets Virtuoso (and replay-from-zero) catch up. We do
    not want a flaky Postgres connection to abort consolidation.

    Returns the event seq, or None if the emit was skipped or failed.
    """
    # Optional dep — see _get_log() for the same lazy-import rationale.
    try:
        from fontem_event_schemas.builders import assert_same_as  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        logger.warning(
            "eventlog: gmr-event-schemas not installed, skipping emit"
        )
        return None
    payload = assert_same_as(
        a_iri=a_iri, b_iri=b_iri,
        confidence=confidence, method=method,
        tier=tier, matched_via_alias=matched_via_alias, rule=rule,
    )
    try:
        seq = await asyncio.to_thread(
            _emit_sync,
            event_type="AssertSameAs",
            iri=a_iri,
            domain=domain,
            payload=payload,
            producer=producer,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "eventlog: AssertSameAs emit failed (a={a}, b={b})",
            a=a_iri, b=b_iri,
        )
        return None
    if seq is not None:
        logger.debug(
            "eventlog: AssertSameAs seq={seq} a={a} b={b}",
            seq=seq, a=a_iri, b=b_iri,
        )
    return seq


# Seven kwargs mirror the RetractSameAs envelope: the pair, why it was
# wrong, who said so, which rule produced it, plus the routing fields.
# The provenance is the point of a retraction — bundling it into a dict
# would hide what a correction is required to record.
async def emit_retract_same_as(  # pylint: disable=too-many-arguments
    *,
    a_iri: str,
    b_iri: str,
    reason: str,
    reviewer: str | None = None,
    retracted_method: str | None = None,
    domain: str = "consolidation",
    producer: str = "fontem-consolidator",
) -> int | None:
    """Emit a RetractSameAs event withdrawing a published equivalence.

    Unlike the assert path, a swallowed failure here leaves a WRONG
    owl:sameAs standing in Virtuoso after an operator has already been
    told it was corrected. The Neo4j side is corrected regardless, so we
    still don't raise — but the seq is returned so the caller can report
    that the retraction did not reach the event log.
    """
    # Optional dep — see _get_log() for the same lazy-import rationale.
    try:
        from fontem_event_schemas.builders import retract_same_as  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        logger.warning(
            "eventlog: fontem-event-schemas not installed, skipping emit"
        )
        return None
    payload = retract_same_as(
        a_iri=a_iri, b_iri=b_iri, reason=reason,
        reviewer=reviewer, retracted_method=retracted_method,
    )
    try:
        seq = await asyncio.to_thread(
            _emit_sync,
            event_type="RetractSameAs",
            iri=a_iri,
            domain=domain,
            payload=payload,
            producer=producer,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "eventlog: RetractSameAs emit failed (a={a}, b={b}) — the "
            "owl:sameAs may still stand in Virtuoso",
            a=a_iri, b=b_iri,
        )
        return None
    logger.info(
        "eventlog: RetractSameAs seq={seq} a={a} b={b} reason={r}",
        seq=seq, a=a_iri, b=b_iri, r=reason,
    )
    return seq


def _emit_many_sync(rows: list[dict], producer: str) -> int:
    """Insert every collected event inside ONE transaction.

    `rows` are the LOGICAL shape actions._emit_same_as_event collects —
    a_iri / b_iri / confidence / method / rule / domain. Building the
    event envelope is this module's job, exactly as it is on the
    single-emit path; the caller has no business knowing the payload
    schema.
    """
    log = _get_log()
    if log is None:
        return 0
    # Optional dep — see _get_log() for the same lazy-import rationale.
    from fontem_event_schemas.builders import assert_same_as  # pylint: disable=import-outside-toplevel
    batch_id = uuid.uuid4()
    with log.batch(batch_id, producer=producer) as emit:
        for r in rows:
            emit.upsert(
                "AssertSameAs",
                iri=r["a_iri"],
                domain=r["domain"],
                payload=assert_same_as(
                    a_iri=r["a_iri"],
                    b_iri=r["b_iri"],
                    confidence=float(r["confidence"]),
                    method=r["method"],
                    rule=r.get("rule"),
                ),
            )
    return len(rows)


async def emit_assert_same_as_many(
    rows: list[dict],
    producer: str = "fontem-consolidator",
) -> int:
    """Emit a consolidation run's AssertSameAs events in one batch.

    One transaction per RUN rather than per decision. The consolidator
    produces 4.5 decisions per consolidation on average and up to 167,
    and EventLog.batch() serialises on a single connection, so emitting
    one at a time made that lock the bottleneck — prod throughput fell
    from 14.5 to ~2.5 events/sec when per-decision emission was turned
    on, a 4.5x drop against a 4.5x rise in transactions.

    Crash safety, which is the reason this can be batched at all:
    the whole consolidation runs inside the trigger's HTTP request, and
    the trigger only advances its offset once that request returns.
    _dispatch calls raise_for_status(), so a consolidator that dies
    mid-run fails the request, the offset is not committed, and the
    event is redelivered. Consolidation is MERGE-based and idempotent,
    so the redo re-derives the same decisions and emits them again.
    A crash therefore costs a repeat, not a lost event.

    What that does leave is a window where Neo4j holds edges the event
    log has not seen yet — the Neo4j writes land per decision, the
    events land at the end. That window closes on the retry. It is the
    same direction of skew the previous code had (Neo4j first, event
    second), just wider, and it never resolves to permanent loss.

    Failures are logged, not raised: the Neo4j write is the immediate
    source of truth and a flaky event store must not abort
    consolidation. Returns the number of events written.
    """
    if not rows:
        return 0
    try:
        return await asyncio.to_thread(_emit_many_sync, rows, producer)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "eventlog: batched AssertSameAs emit failed ({n} events)",
            n=len(rows),
        )
        return 0
