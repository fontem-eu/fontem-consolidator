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

from src.consolidator import actions
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
    from src.consolidator import eventlog

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

    monkeypatch.setattr(eventlog, "_get_log", lambda: _Log())
    rows = [
        {"event_type": "AssertSameAs", "iri": f"http://x/{i}",
         "domain": "company", "payload": {}}
        for i in range(44)
    ]
    n = eventlog._emit_many_sync(rows, producer="test")
    assert n == 44
    assert len(batches) == 1, f"44 decisions opened {len(batches)} transactions"


def test_empty_run_opens_no_transaction(monkeypatch):
    """Most consolidations decide nothing (median 1, many 0). They must not
    pay for a transaction."""
    from src.consolidator import eventlog

    called = []
    monkeypatch.setattr(eventlog, "_get_log", lambda: called.append(1))
    n = asyncio.run(eventlog.emit_assert_same_as_many([]))
    assert n == 0
    assert not called


def test_emit_failure_does_not_abort_consolidation(monkeypatch):
    """The Neo4j write is the immediate source of truth. A flaky event
    store must not take consolidation down with it — the trigger would
    then retry the whole run anyway."""
    from src.consolidator import eventlog

    def _explode(*a, **k):
        raise RuntimeError("events db down")

    monkeypatch.setattr(eventlog, "_emit_many_sync", _explode)
    n = asyncio.run(eventlog.emit_assert_same_as_many(
        [{"event_type": "AssertSameAs", "iri": "http://x",
          "domain": "company", "payload": {}}]))
    assert n == 0
