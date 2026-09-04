"""A consolidation run emits its AssertSameAs events as one batch.

EventLog.batch() serialises on a single connection (fontem-events 0.5.1,
after concurrent use orphaned SAVEPOINTs and filled prod's lock table), so
one transaction per decision made that lock the bottleneck: prod
throughput fell from 14.5 to ~2.5 consolidations/sec when per-decision
emission was enabled — a 4.5x drop against a measured 4.5 decisions per
run.

Batching is safe because the unit of retry is the whole dispatch. The
consolidation runs inside the trigger's HTTP request; _dispatch calls
raise_for_status(); and the consumer commits its offset only after
handle() returns. A consolidator that dies mid-run fails the request, the
offset is not committed, and the event is redelivered — and since
consolidation is MERGE-idempotent the redo re-derives the same decisions
and emits them again. A crash costs a repeat, not a lost event.
"""
# pylint: disable=protected-access,import-outside-toplevel,unused-argument
import asyncio

from src.consolidator import actions, engine, eventlog
from src.consolidator.rules.base import Decision


def _decision(target: str) -> Decision:
    return Decision(
        rule_name="exact_name_country_match",
        action="flag",
        source_id="aaaaaaaa-0000-0000-0000-000000000001",
        target_id=target,
        confidence=0.95,
        entity_type="Company",
        details={},
    )


def test_collector_queues_instead_of_emitting(monkeypatch):
    """With a collector, nothing is written mid-run."""
    emitted = []

    async def _boom(**kwargs):
        emitted.append(kwargs)

    monkeypatch.setattr(actions.eventlog, "emit_assert_same_as", _boom)
    collected: list[dict] = []
    asyncio.run(actions._emit_same_as_event(_decision("bbb"), collected))

    assert not emitted, "event was written immediately despite a collector"
    assert len(collected) == 1
    assert collected[0]["b_iri"].endswith("bbb")
    assert collected[0]["confidence"] == 0.95


def test_without_collector_emits_immediately(monkeypatch):
    """The sweeper and any caller outside a retryable request keep the old
    per-decision behaviour, where each event is durable as it is made."""
    emitted = []

    async def _capture(**kwargs):
        emitted.append(kwargs)

    monkeypatch.setattr(actions.eventlog, "emit_assert_same_as", _capture)
    asyncio.run(actions._emit_same_as_event(_decision("ccc"), None))
    assert len(emitted) == 1


def test_whole_run_is_one_transaction(monkeypatch):
    """The point of the change: N decisions produce ONE batch, not N."""
    batches: list[int] = []

    class _Batch:
        def upsert(self, *a, **k):
            return 1

    class _Log:
        def batch(self, batch_id, producer):
            batches.append(1)
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield _Batch()

            return _cm()

    log = _Log()
    monkeypatch.setattr(eventlog, "_get_log", lambda: log)
    rows = [
        {"a_iri": f"http://x/{i}", "b_iri": f"http://y/{i}",
         "confidence": 0.9, "method": "r", "rule": "r",
         "domain": "company"}
        for i in range(44)
    ]
    n = eventlog._emit_many_sync(rows, producer="test")
    assert n == 44
    assert len(batches) == 1, f"44 decisions opened {len(batches)} transactions"


def test_empty_run_opens_no_transaction(monkeypatch):
    """Most consolidations decide nothing (median 1, many 0). They must not
    pay for a transaction."""
    called = []
    monkeypatch.setattr(eventlog, "_get_log", lambda: called.append(1))
    n = asyncio.run(eventlog.emit_assert_same_as_many([]))
    assert n == 0
    assert not called


def test_emit_failure_does_not_abort_consolidation(monkeypatch):
    """The Neo4j write is the immediate source of truth. A flaky event
    store must not take consolidation down with it — the trigger would
    then retry the whole run anyway."""
    def _explode(*a, **k):
        raise RuntimeError("events db down")

    monkeypatch.setattr(eventlog, "_emit_many_sync", _explode)
    n = asyncio.run(eventlog.emit_assert_same_as_many(
        [{"a_iri": "http://x", "b_iri": "http://y",
          "confidence": 0.9, "method": "r", "rule": "r",
          "domain": "company"}]))
    assert n == 0


def test_batch_rows_are_the_shape_the_collector_produces(monkeypatch):
    """The bug this file failed to catch.

    _emit_many_sync used to read r["event_type"] / r["iri"] / r["payload"],
    but actions._emit_same_as_event collects a_iri / b_iri / confidence /
    method / rule / domain. Every batched emit raised KeyError, the broad
    except in emit_assert_same_as_many swallowed it, and the function
    returned 0 — so the batched path emitted NOTHING, silently, for as
    long as it existed.

    It survived because the old tests fabricated rows in the shape the
    consumer wanted rather than the shape the producer sends. This one
    builds the row through the real collector.
    """

    collected: list[dict] = []
    decision = Decision(
        rule_name="exact_lei_match", action="merge", source_id="A",
        target_id="B", confidence=0.99, entity_type="Company", details={},
    )
    asyncio.run(actions._emit_same_as_event(decision, collected))
    assert collected, "collector produced no row"

    seen: list[tuple] = []

    class _Batch:
        def upsert(self, event_type, *, iri, domain, payload):
            seen.append((event_type, iri, domain, payload))

    class _Log:
        def batch(self, batch_id, producer):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield _Batch()

            return _cm()

    monkeypatch.setattr(eventlog, "_get_log", _Log)
    n = eventlog._emit_many_sync(collected, producer="test")

    assert n == 1
    event_type, iri, domain, payload = seen[0]
    assert event_type == "AssertSameAs"
    assert iri == "http://data.fontem.eu/id/Company/A"
    assert domain == "company"
    assert payload["a_iri"] == "http://data.fontem.eu/id/Company/A"
    assert payload["b_iri"] == "http://data.fontem.eu/id/Company/B"
    assert payload["method"] == "exact_lei_match"


def test_batched_payload_validates_against_the_schema():
    """A payload the schema rejects is as lost as one never emitted."""
    from jsonschema import validate

    from fontem_event_schemas.loader import load_schema


    collected: list[dict] = []
    decision = Decision(
        rule_name="exact_lei_match", action="merge", source_id="A",
        target_id="B", confidence=0.99, entity_type="Company", details={},
    )
    asyncio.run(actions._emit_same_as_event(decision, collected))

    from fontem_event_schemas.builders import assert_same_as
    payload = assert_same_as(
        a_iri=collected[0]["a_iri"], b_iri=collected[0]["b_iri"],
        confidence=collected[0]["confidence"], method=collected[0]["method"],
        rule=collected[0]["rule"],
    )
    validate(instance=payload, schema=load_schema("AssertSameAs"))


def test_pairs_are_marked_only_after_the_events_land(monkeypatch):
    """Emit first, mark second, mark nothing on a partial batch.

    The event is the only record an assertion exists — Neo4j holds no
    equivalences — so marking before the emit would let a dropped batch
    look permanently settled and never retry. That is silent, permanent
    data loss dressed up as success.
    """
    pending = [{
        "a_iri": "http://x", "b_iri": "http://y", "confidence": 0.9,
        "method": "r", "rule": "r", "domain": "company",
        "entity_type": "Company", "source_id": "A", "target_id": "B",
    }]

    marked: list[int] = []

    async def _mark(_driver, _db, rows):
        marked.append(len(rows))
        return len(rows)

    monkeypatch.setattr(actions, "mark_asserted", _mark)

    # Happy path: everything emitted, so everything is marked.
    async def _all(_rows):
        return len(_rows)

    monkeypatch.setattr(engine.eventlog, "emit_assert_same_as_many", _all)
    asyncio.run(engine._flush_pending_events(None, "neo4j", "run-1", list(pending)))
    assert marked == [1]

    # Failed batch: nothing is marked, so the pair re-derives next sweep.
    marked.clear()

    async def _none(_rows):
        return 0

    monkeypatch.setattr(engine.eventlog, "emit_assert_same_as_many", _none)
    asyncio.run(engine._flush_pending_events(None, "neo4j", "run-2", list(pending)))
    assert not marked, "a lost batch must not be recorded as settled"
