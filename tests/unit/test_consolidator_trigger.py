"""Tests for ConsolidatorTrigger — the gmr-events consumer that
POSTs each input event to the consolidator dispatch endpoint.

Covers the cycle-prevention contract:
  - Upsert/Delete events are dispatched.
  - AssertSameAs / RetractSameAs / Begin/EndGraphReplace are skipped
    (the trigger MUST NOT re-dispatch the consolidator's own outputs).
  - 409 from the dispatch is treated as success.
  - Other 4xx/5xx raise so the EventConsumer's batch retry handles it.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest


def _envelope(seq: int, event_type: str, **overrides):
    """Build a duck-typed envelope. ConsolidatorTrigger reads
    seq/event_type/iri/domain/payload/batch_id/producer."""
    base = {
        "seq": seq, "event_type": event_type, "iri": "x", "domain": "test",
        "payload": {"gmr_id": "abc"} if event_type.startswith("Upsert") else {},
        "batch_id": None, "producer": "test",
    }
    base.update(overrides)
    return type("E", (), base)()


@pytest.fixture
def trigger():
    """Construct the trigger with env stubbed; bypass EventConsumer's
    Postgres-touching __init__ pieces by passing a config directly."""
    with patch.dict(os.environ, {
        "CONSOLIDATOR_URL": "http://fontem-consolidator.test:8000",
        "EVENT_CONSUMER_NAME": "consolidator_trigger",
        "EVENTS_DATABASE_URL": "postgresql://stub",
    }):
        from gmr_events.consumer import ConsumerConfig
        from src.consolidator.trigger.consumer import ConsolidatorTrigger
        cfg = ConsumerConfig(
            name="consolidator_trigger",
            dsn="postgresql://stub",
            metrics_port=None,
        )
        yield ConsolidatorTrigger(cfg)


def _client_factory(handler):
    """Build a (constructor) callable that pretends to be httpx.Client
    but returns a real client with a mock transport. The trigger calls
    ``httpx.Client(timeout=...)``; our factory ignores that and returns
    a transport-stubbed client instead."""
    real_client_cls = httpx.Client

    def _ctor(*_args, **_kwargs):
        return real_client_cls(transport=httpx.MockTransport(handler))

    return _ctor


def test_input_types_dispatched(trigger):
    batch = [
        _envelope(10, "UpsertCompany"),
        _envelope(11, "UpsertFiling", payload={"gmr_id": "abc"}),
        _envelope(12, "DeleteCompany"),
    ]
    posted: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        posted.append(str(req.url))
        return httpx.Response(200, json={"outcome": "consolidated"})

    with patch(
        "src.consolidator.trigger.consumer.httpx.Client",
        _client_factory(handler),
    ):
        trigger.handle(batch)
    assert len(posted) == 3, posted
    assert all(url.endswith("/events/dispatch") for url in posted)


def test_consolidator_outputs_skipped_no_loop(trigger):
    """The cycle-break: if the consolidator emits AssertSameAs back
    into the log, the trigger must NOT POST it back to dispatch.
    Same for RetractSameAs and the Begin/End graph-replace markers."""
    batch = [
        _envelope(20, "AssertSameAs",   payload={"a_iri": "x", "b_iri": "y"}),
        _envelope(21, "RetractSameAs",  payload={"a_iri": "x", "b_iri": "y"}),
        _envelope(22, "BeginGraphReplace", payload={"graph_iri": "g"}),
        _envelope(23, "EndGraphReplace",   payload={"graph_iri": "g"}),
    ]
    posted: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        posted.append(str(req.url))
        return httpx.Response(200)

    with patch(
        "src.consolidator.trigger.consumer.httpx.Client",
        _client_factory(handler),
    ):
        trigger.handle(batch)
    assert posted == [], (
        "trigger dispatched a consolidator-output event; loop is open"
    )


def test_409_is_treated_as_success(trigger):
    batch = [_envelope(30, "UpsertCompany")]

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "already done"})

    with patch(
        "src.consolidator.trigger.consumer.httpx.Client",
        _client_factory(handler),
    ):
        # No exception → batch consumed, offset advances.
        trigger.handle(batch)


def test_5xx_raises_for_retry(trigger):
    batch = [_envelope(40, "UpsertCompany")]

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="db down")

    with patch(
        "src.consolidator.trigger.consumer.httpx.Client",
        _client_factory(handler),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            trigger.handle(batch)


def test_4xx_other_than_409_raises(trigger):
    """A 400 means the trigger sent malformed data — we want
    EventConsumer's batch retry → DLQ to surface it, not silently skip."""
    batch = [_envelope(50, "UpsertCompany")]

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad payload")

    with patch(
        "src.consolidator.trigger.consumer.httpx.Client",
        _client_factory(handler),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            trigger.handle(batch)


def test_concurrent_dispatch_threadpool(trigger):
    """The bounded executor sends the whole batch through the
    pool. We don't pin call ORDER (it's parallel) — just that all
    events reach the dispatch endpoint."""
    import threading
    import time as _t
    trigger.concurrency = 4
    batch = [_envelope(60 + i, "UpsertCompany") for i in range(8)]
    seen_threads: set[str] = set()
    posted: list[int] = []
    lock = threading.Lock()

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.read()
        import json
        seq = json.loads(body).get("seq")
        # Hold the worker briefly so the executor actually has to
        # use multiple threads to drain the batch — without this
        # delay an instant-return mock can serialise everything
        # onto a single worker on under-loaded CI runners.
        _t.sleep(0.05)
        with lock:
            posted.append(seq)
            seen_threads.add(threading.current_thread().name)
        return httpx.Response(200, json={"outcome": "consolidated"})

    with patch(
        "src.consolidator.trigger.consumer.httpx.Client",
        _client_factory(handler),
    ):
        trigger.handle(batch)
    assert sorted(posted) == sorted(60 + i for i in range(8))
    # With concurrency=4 and 8 events that each hold for 50ms we
    # expect at least 2 distinct worker threads. Single-CPU CI is
    # still timesharing, but the explicit sleep forces overlap so
    # the pool MUST spin up >1 worker.
    assert len(seen_threads) >= 2, (
        f"expected concurrent dispatch, all ran on {seen_threads}"
    )
