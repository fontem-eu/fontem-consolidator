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
_log_singleton = None  # type: ignore[var-annotated]


def _get_log():
    """Return the lazily-constructed fontem_events EventLog (or None
    if the env isn't configured)."""
    global _log_singleton
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
        try:
            from fontem_events import EventLog  # local import: optional dep
        except ImportError:  # pragma: no cover
            logger.warning(
                "eventlog: gmr-events not installed, emit will no-op"
            )
            return None
        _log_singleton = EventLog.from_env()
        logger.info("eventlog: connected to events database")
        return _log_singleton


def _emit_sync(
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


async def emit_assert_same_as(
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
    try:
        from fontem_event_schemas.builders import assert_same_as
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
