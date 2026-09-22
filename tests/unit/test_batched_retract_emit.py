"""RetractSameAs in batches, for the C1 cleanup.

Mirrors test_batched_same_as_emit: one transaction per batch, the
payload built by the schema's own builder, all-or-nothing so the caller
can tell a dropped batch from a landed one.
"""
# pylint: disable=protected-access,import-outside-toplevel,unused-argument
import asyncio
from contextlib import contextmanager

from src.consolidator import eventlog


def _rows(n):
    return [
        {"a_iri": f"http://data.fontem.eu/id/Company/junk-{i}",
         "b_iri": f"http://data.fontem.eu/id/Company/other-{i}",
         "reason": "junk name", "reviewer": "retract_junk_names",
         "retracted_method": "fuzzy_name_same_country", "domain": "company"}
        for i in range(n)
    ]


def _log(batches, seen):
    class _Batch:
        def upsert(self, event_type, *, iri, domain, payload):
            seen.append((event_type, iri, domain, payload))
            return len(seen)

    class _Log:
        def batch(self, batch_id, producer):
            batches.append(producer)

            @contextmanager
            def _cm():
                yield _Batch()

            return _cm()

    return _Log()


def test_a_batch_is_one_transaction(monkeypatch):
    batches, seen = [], []
    monkeypatch.setattr(eventlog, "_get_log", lambda: _log(batches, seen))
    n = eventlog._emit_retract_many_sync(_rows(37), producer="c1")
    assert n == 37
    assert batches == ["c1"], f"37 retractions opened {len(batches)} transactions"
    assert len(seen) == 37


def test_the_envelope_is_the_single_emit_shape(monkeypatch):
    """Same event type, routing and builder as emit_retract_same_as, so
    the sinks cannot tell a batched retraction from an operator's."""
    seen = []
    monkeypatch.setattr(eventlog, "_get_log", lambda: _log([], seen))
    eventlog._emit_retract_many_sync(_rows(1), producer="c1")
    event_type, iri, domain, payload = seen[0]
    assert event_type == "RetractSameAs"
    assert iri == "http://data.fontem.eu/id/Company/junk-0"
    assert domain == "company"
    assert payload == {
        "a_iri": "http://data.fontem.eu/id/Company/junk-0",
        "b_iri": "http://data.fontem.eu/id/Company/other-0",
        "reason": "junk name",
        "reviewer": "retract_junk_names",
        "retracted_method": "fuzzy_name_same_country",
    }


def test_the_payload_validates_against_the_schema(monkeypatch):
    from jsonschema import validate

    from fontem_event_schemas.loader import load_schema

    seen = []
    monkeypatch.setattr(eventlog, "_get_log", lambda: _log([], seen))
    row = _rows(1)[0]
    row["reviewer"] = None  # optional fields are dropped, not sent as null
    eventlog._emit_retract_many_sync([row], producer="c1")
    payload = seen[0][3]
    assert "reviewer" not in payload
    validate(instance=payload, schema=load_schema("RetractSameAs"))


def test_no_event_store_means_nothing_landed(monkeypatch):
    monkeypatch.setattr(eventlog, "_get_log", lambda: None)
    assert eventlog._emit_retract_many_sync(_rows(3), producer="c1") == 0


def test_an_empty_batch_opens_no_transaction(monkeypatch):
    called = []
    monkeypatch.setattr(eventlog, "_get_log", lambda: called.append(1))
    assert asyncio.run(eventlog.emit_retract_same_as_many([])) == 0
    assert not called


def test_a_failed_batch_reports_zero_rather_than_raising(monkeypatch):
    """The caller records :NOT_SAME_AS only on a full batch; a raise
    would take the whole Job down, a silent success would leave a wrong
    :SAME_AS standing behind a correction that says it was withdrawn."""
    def _explode(*a, **k):
        raise RuntimeError("events db down")

    monkeypatch.setattr(eventlog, "_emit_retract_many_sync", _explode)
    assert asyncio.run(eventlog.emit_retract_same_as_many(_rows(2))) == 0


def test_a_landed_batch_reports_its_size(monkeypatch):
    monkeypatch.setattr(eventlog, "_emit_retract_many_sync", lambda rows, producer: len(rows))
    assert asyncio.run(eventlog.emit_retract_same_as_many(_rows(5))) == 5
