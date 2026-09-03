"""The manual-review loop: approvals publish, rejections stick.

Two defects this covers, both of which made the review queue
non-functional rather than merely incomplete:

1. A reviewer's "merge" collapsed the nodes but emitted no
   AssertSameAs, so an APPROVED equivalence never reached Virtuoso
   and the absorbed node's IRI dangled.

2. A reviewer's "reject" only DELETEd the :SAME_AS edge. The rules are
   deterministic and the sweeper re-runs them over every entity
   forever, so the very next pass MERGEd the pair straight back. No
   rejection could ever be made to stick and the queue could never be
   drained. :NOT_SAME_AS is the durable veto.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _stub_driver(edge_record):
    session = AsyncMock()
    statements: list[str] = []

    class _Result:
        def __init__(self, rec):
            self._rec = rec

        async def single(self):
            return self._rec

        def __aiter__(self):
            async def _gen():
                if self._rec is not None:
                    yield self._rec
            return _gen()

    async def run(query, *args, **kwargs):  # pylint: disable=unused-argument
        statements.append(query)
        # Only the first query (the edge lookup) returns a record.
        return _Result(edge_record if len(statements) == 1 else None)

    session.run = AsyncMock(side_effect=run)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    driver = AsyncMock()
    driver.session = MagicMock(return_value=ctx)
    return driver, statements


@pytest.fixture(name="edge")
def _edge():
    rel = {"method": "fuzzy_name_same_country", "confidence": 0.94}
    return {
        "a": {"gmr_id": "A"},
        "b": {"gmr_id": "B"},
        "r": rel,
        "label": "Company",
    }


def _decide(c, driver, decision, emit):
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)), \
         patch("src.api.routes.candidates.eventlog.emit_assert_same_as", emit):
        return c.post(
            "/candidates/A/B/decide",
            json={"decision": decision, "reviewer": "someone@fontem.eu"},
        )


def test_approved_merge_publishes_owl_same_as(client, edge):
    c, _ = client
    driver, _stmts = _stub_driver(edge)
    emit = AsyncMock(return_value=4242)
    r = _decide(c, driver, "merge", emit)

    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "manual_merge"
    assert body["event_seq"] == 4242
    assert body["projected"] is True

    emit.assert_awaited_once()
    kwargs = emit.await_args.kwargs
    assert kwargs["a_iri"] == "http://data.fontem.eu/id/Company/A"
    assert kwargs["b_iri"] == "http://data.fontem.eu/id/Company/B"


def test_failed_emit_is_reported_not_swallowed(client, edge):
    """emit_assert_same_as absorbs its own failures and returns None.
    The merge already happened, so the request still succeeds — but a
    silently unpublished approval is invisible, so say so."""
    c, _ = client
    driver, _stmts = _stub_driver(edge)
    emit = AsyncMock(return_value=None)
    r = _decide(c, driver, "merge", emit)

    assert r.status_code == 200
    assert r.json()["projected"] is False


def test_reject_records_durable_veto(client, edge):
    c, _ = client
    driver, stmts = _stub_driver(edge)
    emit = AsyncMock()
    r = _decide(c, driver, "reject", emit)

    assert r.status_code == 200
    assert r.json()["outcome"] == "manual_reject"
    written = " ".join(stmts)
    assert "NOT_SAME_AS" in written, "rejection must survive the next sweep"
    assert "DELETE r" in written
    emit.assert_not_awaited()


def test_keep_as_related_moves_off_same_as(client, edge):
    """'Keep as related' means NOT the same entity. Leaving it on a
    reviewed :SAME_AS edge fed gds/wcc_collapse, which projects exactly
    `reviewed=true AND conflict=false` and merges with force_auto_merge
    — so the answer 'these are merely related' would have deleted a
    node."""
    c, _ = client
    driver, stmts = _stub_driver(edge)
    emit = AsyncMock()
    r = _decide(c, driver, "keep_as_related", emit)

    assert r.status_code == 200
    assert r.json()["outcome"] == "manual_keep_related"
    written = " ".join(stmts)
    assert "RELATED_TO" in written
    assert "NOT_SAME_AS" in written
    assert "DELETE r" in written
    assert "r.reviewed = true" not in written
    emit.assert_not_awaited()
