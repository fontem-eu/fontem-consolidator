"""The decision log lives in Postgres and records only what changed something.

It used to be three Neo4j nodes per rule evaluation — 261M of prod's
277.8M nodes — most of them the sweeper re-recording a proposal it had
already made. These pin what is written, where, and what is not.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from src.consolidator import actions, audit
from src.consolidator.neo4j import migrations
from src.consolidator.rules.base import Candidate, Decision, Entity


class _Cursor:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, rows=None, rowcounts=None):
        self.executed: list[tuple[str, object]] = []
        self._rows = rows or []
        self._rowcounts = list(rowcounts or [])

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        rowcount = self._rowcounts.pop(0) if self._rowcounts else 0
        return _Cursor(self._rows, rowcount)


class _Store:
    def __init__(self, conn=None, fail=False):
        self.conn = conn or _Conn()
        self.fail = fail

    @staticmethod
    def configured():
        return True

    def run(self, fn):
        if self.fail:
            raise RuntimeError("postgres is down")
        return fn(self.conn)


@pytest.fixture(name="store")
def _store(monkeypatch):
    store = _Store()
    monkeypatch.setattr(audit, "_store", store)
    return store


def _decision(**over):
    fields = {"rule_name": "fuzzy_name_same_country", "action": "flag", "source_id": "A",
              "target_id": "B", "confidence": 0.93, "entity_type": "Company",
              "details": {"score": 0.93}}
    fields.update(over)
    return Decision(**fields)


def _candidate(entity_id="B"):
    return Candidate(entity=Entity("Company", entity_id, {}), context={})


def _record(outcome):
    return asyncio.run(audit.record_decision(
        run_id="3f2b8f0e-1c1e-4c55-9b0a-2d0f7d2b9a11", decision=_decision(),
        decision_type=outcome, candidate=_candidate()))


@pytest.mark.parametrize("outcome", ["noop", "reflag", "enrich"])
def test_a_decision_that_changed_nothing_is_not_recorded(store, outcome):
    assert _record(outcome) is None
    assert not store.conn.executed


@pytest.mark.parametrize("outcome", sorted(audit.RECORDED_OUTCOMES))
def test_a_decision_that_changed_something_is_recorded_as_a_rule_row(store, outcome):
    decision_id = _record(outcome)
    (sql, row), = store.conn.executed
    assert "INSERT INTO consolidation.decision_log" in sql
    assert row["decision_id"] == decision_id and row["origin"] == "rule"
    assert row["decision_type"] == outcome and row["rule_name"] == "fuzzy_name_same_country"
    assert (row["source_id"], row["target_id"]) == ("A", "B")
    assert json.loads(row["details"]) == {"score": 0.93}


def test_a_rule_record_that_fails_does_not_stop_consolidation(monkeypatch):
    monkeypatch.setattr(audit, "_store", _Store(fail=True))
    assert _record("flag") is None


def test_a_human_decision_is_recorded_and_a_failure_is_raised(store, monkeypatch):
    decision_id = asyncio.run(audit.record_review(
        decision_type="manual_merge", rule_name="fuzzy_name_same_country",
        entity_type="Company", source_id="A", target_id="B", reviewer="gonçalo",
        review_note="same VAT on the invoice"))
    (_sql, row), = store.conn.executed
    assert row["decision_id"] == decision_id and row["origin"] == "human"
    assert row["reviewer"] == "gonçalo" and row["review_note"] == "same VAT on the invoice"

    monkeypatch.setattr(audit, "_store", _Store(fail=True))
    with pytest.raises(RuntimeError):
        asyncio.run(audit.record_review(
            decision_type="manual_reject", rule_name=None, entity_type="Company",
            source_id="A", target_id="B", reviewer="gonçalo"))


def test_listing_without_filters_has_no_where_clause(store):
    asyncio.run(audit.list_decisions(limit=100))
    (sql, params), = store.conn.executed
    assert "WHERE" not in sql and "ORDER BY decided_at DESC LIMIT %(limit)s" in sql
    assert params == {"limit": 100}


def test_listing_applies_one_clause_per_filter(store):
    asyncio.run(audit.list_decisions(
        entity_type="Company", entity_id="abc", rule_name="fuzzy",
        decision_type="flag", since="2026-01-01", cursor="2026-04-01", limit=25))
    (sql, params), = store.conn.executed
    for clause in ("entity_type = %(entity_type)s", "rule_name = %(rule_name)s",
                   "decision_type = %(decision_type)s",
                   "(source_id = %(entity_id)s OR target_id = %(entity_id)s)",
                   "decided_at >= %(since)s", "decided_at < %(cursor)s"):
        assert clause in sql, clause
    assert params["limit"] == 25 and params["entity_id"] == "abc"


def test_pruning_deletes_only_rule_decisions_in_batches(store):
    store.conn = _Conn(rowcounts=[3, 3, 1])
    assert asyncio.run(audit.prune(180, batch=3)) == 7
    assert len(store.conn.executed) == 3
    assert all("origin = 'rule'" in sql for sql, _ in store.conn.executed)


def test_nothing_is_written_without_the_events_database(monkeypatch):
    monkeypatch.delenv("EVENTS_DATABASE_URL", raising=False)
    monkeypatch.setattr(audit, "_store", audit._Store())  # pylint: disable=protected-access
    assert _record("flag") is None
    assert asyncio.run(audit.list_decisions()) == []
    assert asyncio.run(audit.prune(180)) == 0


def test_startup_creates_no_audit_indexes_in_the_graph():
    assert not [s for s in migrations.INDEX_CYPHER
                if "DecisionLog" in s or "ConsolidationRun" in s or "RuleApplication" in s]


@pytest.mark.parametrize("change, outcome", [
    (None, "noop"), ("unchanged", "reflag"), ("new", "flag"), ("changed", "flag"),
])
def test_a_proposal_says_whether_it_changed(change, outcome):
    """The sweeper re-proposes every pending pair each rotation; only a new
    or changed proposal is a decision worth recording."""
    with patch.object(actions, "_propose_candidate", new=AsyncMock(return_value=change)):
        got = asyncio.run(actions.execute(None, "neo4j", decision=_decision()))
    assert got == outcome
