"""What the consolidator decided, and who decided it — kept in Postgres.

Until 2026-09 every rule evaluation wrote a :ConsolidationRun, a
:RuleApplication and a :DecisionLog into Neo4j: 261M of prod's 277.8M
nodes, growing by 3.7M runs a day, mostly the sweeper recording the same
conclusion again. In a sample of 200k fuzzy-match decisions 30% were
noops and one decision had been logged 1,212 times. The graph holds what
the site serves; how it was consolidated is an operational record, and
it lives in Postgres beside the event log (``EVENTS_DATABASE_URL``).

Only what changed something is recorded: an assertion, a link, a new or
changed proposal (``RECORDED_OUTCOMES``), and every human decision. A
rule finding nothing ("noop") or re-confirming a proposal it already made
("reflag") is counted in Prometheus, not written. Rule decisions are
pruned after ``decision_retention_days``; human decisions are kept.

Rule decisions are best effort: a Postgres failure is logged and the
consolidation carries on, as for the event emits. A human decision that
cannot be recorded fails the request, so the reviewer knows to retry.

psycopg's sync API runs through ``asyncio.to_thread`` on one shared,
lazily opened connection, like the event-log shim. Without
``EVENTS_DATABASE_URL`` (unit tests, dev) nothing is written.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from src.consolidator.rules.base import Candidate, Decision

#: Outcomes a rule decision is recorded for: each one changed the graph
#: or published something. See actions.execute for the full set.
RECORDED_OUTCOMES = frozenset({"auto_assert", "auto_link", "flag", "conflict"})

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS consolidation;
CREATE TABLE IF NOT EXISTS consolidation.decision_log (
    decision_id   uuid PRIMARY KEY,
    decided_at    timestamptz NOT NULL DEFAULT now(),
    origin        text NOT NULL CHECK (origin IN ('rule', 'human')),
    run_id        uuid,
    decision_type text NOT NULL,
    rule_name     text,
    entity_type   text,
    source_id     text,
    target_id     text,
    confidence    double precision,
    reviewer      text,
    review_note   text,
    details       jsonb
);
CREATE INDEX IF NOT EXISTS decision_log_entity
    ON consolidation.decision_log (entity_type, source_id);
CREATE INDEX IF NOT EXISTS decision_log_target
    ON consolidation.decision_log (target_id);
CREATE INDEX IF NOT EXISTS decision_log_decided_at
    ON consolidation.decision_log (decided_at);
CREATE INDEX IF NOT EXISTS decision_log_rule
    ON consolidation.decision_log (rule_name, decided_at);
"""

_INSERT = """
INSERT INTO consolidation.decision_log
  (decision_id, decided_at, origin, run_id, decision_type, rule_name,
   entity_type, source_id, target_id, confidence, reviewer, review_note, details)
VALUES
  (%(decision_id)s, %(decided_at)s, %(origin)s, %(run_id)s, %(decision_type)s,
   %(rule_name)s, %(entity_type)s, %(source_id)s, %(target_id)s,
   %(confidence)s, %(reviewer)s, %(review_note)s, %(details)s)
"""

_COLUMNS = ("decision_id", "decided_at", "origin", "run_id", "decision_type",
            "rule_name", "entity_type", "source_id", "target_id", "confidence",
            "reviewer", "review_note", "details")


class _Store:
    """One connection, serialised by a lock, reopened when it drops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = None
        self._schema_ready = False

    @staticmethod
    def configured() -> bool:
        return bool(os.environ.get("EVENTS_DATABASE_URL"))

    def run(self, fn):
        """Call ``fn(conn)`` under the lock; reconnect once if it dropped."""
        import psycopg  # pylint: disable=import-outside-toplevel
        with self._lock:
            for attempt in (1, 2):
                conn = self._connection(psycopg)
                try:
                    return fn(conn)
                except psycopg.OperationalError:
                    self._conn = None
                    if attempt == 2:
                        raise
        return None  # pragma: no cover  (loop always returns or raises)

    def _connection(self, psycopg):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(os.environ["EVENTS_DATABASE_URL"], autocommit=True)
            self._schema_ready = False
        if not self._schema_ready:
            try:
                self._conn.execute(SCHEMA_SQL)
            except (psycopg.errors.UniqueViolation, psycopg.errors.DuplicateObject,
                    psycopg.errors.DuplicateTable):
                # Another pod created it in the same instant; it exists now.
                pass
            self._schema_ready = True
        return self._conn


_store = _Store()


def _row(**values: Any) -> dict:
    row = dict.fromkeys(_COLUMNS)
    row.update(values)
    row["decision_id"] = row["decision_id"] or str(uuid4())
    row["decided_at"] = row["decided_at"] or datetime.now(timezone.utc)
    if row["details"] is not None:
        row["details"] = json.dumps(row["details"], default=str)
    return row


def _insert(row: dict) -> None:
    _store.run(lambda conn: conn.execute(_INSERT, row))


# The kwargs are the columns of the decision record; bundling them would
# only move the same names into a struct.
async def record_decision(  # pylint: disable=too-many-arguments
    *,
    run_id: str,
    decision: Decision,
    decision_type: str,
    candidate: Candidate,
) -> str | None:
    """Record a rule's decision when it changed something.

    ``decision_type`` is what the executor applied (actions.execute).
    Returns the decision id, or None when it was not recorded.
    """
    if decision_type not in RECORDED_OUTCOMES or not _store.configured():
        return None
    details = dict(decision.details or {})
    if candidate.entity.id != decision.target_id:
        details["candidate_id"] = candidate.entity.id
    row = _row(origin="rule", run_id=run_id, decision_type=decision_type,
               rule_name=decision.rule_name, entity_type=decision.entity_type,
               source_id=decision.source_id, target_id=decision.target_id,
               confidence=decision.confidence, details=details or None)
    try:
        await asyncio.to_thread(_insert, row)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("audit: decision {rule} {src}->{dst} not recorded: {exc}",
                       rule=decision.rule_name, src=decision.source_id,
                       dst=decision.target_id, exc=exc)
        return None
    return row["decision_id"]


# Same columns as record_decision, for a person's decision.
async def record_review(  # pylint: disable=too-many-arguments
    *,
    decision_type: str,
    rule_name: str | None,
    entity_type: str,
    source_id: str,
    target_id: str,
    reviewer: str,
    review_note: str | None = None,
    confidence: float | None = None,
) -> str:
    """Record a human decision. Raises if it cannot be written."""
    row = _row(origin="human", decision_type=decision_type, rule_name=rule_name,
               entity_type=entity_type, source_id=source_id, target_id=target_id,
               confidence=confidence, reviewer=reviewer, review_note=review_note)
    if not _store.configured():
        logger.warning("audit: EVENTS_DATABASE_URL unset; review {t} {src}->{dst} not recorded",
                       t=decision_type, src=source_id, dst=target_id)
        return row["decision_id"]
    await asyncio.to_thread(_insert, row)
    return row["decision_id"]


# Filters of GET /decisions, one per query parameter.
async def list_decisions(  # pylint: disable=too-many-arguments
    *,
    entity_type: str | None = None,
    entity_id: str | None = None,
    rule_name: str | None = None,
    decision_type: str | None = None,
    since: str | None = None,
    cursor: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Newest first. ``cursor`` is the decided_at of the last row seen."""
    if not _store.configured():
        return []
    where, params = [], {"limit": limit}
    for column, value in (("entity_type", entity_type), ("rule_name", rule_name),
                          ("decision_type", decision_type)):
        if value:
            where.append(f"{column} = %({column})s")
            params[column] = value
    if entity_id:
        where.append("(source_id = %(entity_id)s OR target_id = %(entity_id)s)")
        params["entity_id"] = entity_id
    if since:
        where.append("decided_at >= %(since)s")
        params["since"] = since
    if cursor:
        where.append("decided_at < %(cursor)s")
        params["cursor"] = cursor
    sql = (f"SELECT {', '.join(_COLUMNS)} FROM consolidation.decision_log "
           f"{'WHERE ' + ' AND '.join(where) if where else ''} "
           "ORDER BY decided_at DESC LIMIT %(limit)s")

    def _select(conn) -> list[dict]:
        cur = conn.execute(sql, params)
        return [_jsonable(dict(zip(_COLUMNS, r))) for r in cur.fetchall()]

    return await asyncio.to_thread(_store.run, _select)


def _jsonable(row: dict) -> dict:
    row["decision_id"] = str(row["decision_id"])
    row["run_id"] = str(row["run_id"]) if row["run_id"] else None
    row["decided_at"] = row["decided_at"].isoformat()
    return row


async def prune(retention_days: int, *, batch: int = 50_000) -> int:
    """Delete rule decisions older than the retention; keep human ones."""
    if not _store.configured():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    def _delete(conn) -> int:
        total = 0
        while True:
            cur = conn.execute(
                "DELETE FROM consolidation.decision_log WHERE decision_id IN ("
                " SELECT decision_id FROM consolidation.decision_log"
                " WHERE origin = 'rule' AND decided_at < %s LIMIT %s)",
                (cutoff, batch))
            total += cur.rowcount
            if cur.rowcount < batch:
                return total

    deleted = await asyncio.to_thread(_store.run, _delete)
    if deleted:
        logger.info("audit: pruned {n} rule decisions older than {d} days",
                    n=deleted, d=retention_days)
    return deleted
