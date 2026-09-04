"""The review loop: proposals, approvals, declines, and corrections.

The model these tests pin down:

  :SAME_AS_CANDIDATE  a rule's proposal. Emits nothing.
  :SAME_AS            an approved equivalence. Emits AssertSameAs.
  :NOT_SAME_AS        a correction of something already asserted.
                      Emits RetractSameAs.

The distinction that keeps being got wrong is between DECLINING and
CORRECTING. Declining rejects a proposal that was never published, so
there is nothing to retract and no event is emitted. Correcting
withdraws a claim the platform actually made, so it must emit — and it
can only work because approval leaves both nodes alive.
"""

from unittest.mock import AsyncMock, MagicMock, patch


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


def _edge(status="pending"):
    return {
        "a": {"gmr_id": "A"},
        "b": {"gmr_id": "B"},
        "r": {
            "method": "fuzzy_name_same_country",
            "confidence": 0.94,
            "status": status,
        },
        "label": "Company",
    }


def _decide(c, driver, decision, emit, asserted=True):
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)), \
         patch("src.api.routes.candidates.eventlog.emit_assert_same_as", emit), \
         patch("src.api.routes.candidates.actions.assert_same_as",
               AsyncMock(return_value=asserted)):
        return c.post(
            "/candidates/A/B/decide",
            json={"decision": decision, "reviewer": "someone@fontem.eu"},
        )


# --------------------------------------------------------------- approve


def test_approval_asserts_and_publishes(client):
    c, _ = client
    driver, _stmts = _stub_driver(_edge())
    emit = AsyncMock(return_value=4242)
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)), \
         patch("src.api.routes.candidates.eventlog.emit_assert_same_as", emit), \
         patch("src.api.routes.candidates.actions.assert_same_as",
               AsyncMock(return_value=True)) as asserter:
        r = c.post("/candidates/A/B/decide",
                   json={"decision": "approve", "reviewer": "someone@fontem.eu"})

    assert r.status_code == 200
    body = r.json()
    assert body["event_seq"] == 4242
    assert body["projected"] is True

    asserter.assert_awaited_once()
    assert asserter.await_args.kwargs["origin"] == "review"
    emit.assert_awaited_once()
    kwargs = emit.await_args.kwargs
    assert kwargs["a_iri"] == "http://data.fontem.eu/id/Company/A"
    assert kwargs["b_iri"] == "http://data.fontem.eu/id/Company/B"


def test_legacy_merge_value_still_means_approve(client):
    """The web app ships separately; don't require a simultaneous deploy."""
    c, _ = client
    driver, _stmts = _stub_driver(_edge())
    emit = AsyncMock(return_value=1)
    r = _decide(c, driver, "merge", emit)
    assert r.status_code == 200
    emit.assert_awaited_once()


def test_approval_blocked_by_existing_correction(client):
    """A :NOT_SAME_AS outranks a later approval, and nothing is published."""
    c, _ = client
    driver, _stmts = _stub_driver(_edge())
    emit = AsyncMock()
    r = _decide(c, driver, "approve", emit, asserted=False)
    assert r.status_code == 409
    emit.assert_not_awaited()


def test_failed_emit_is_reported_not_swallowed(client):
    """emit_assert_same_as absorbs its failures and returns None. The
    :SAME_AS edge exists, so the request succeeds — but an approval that
    never reached Virtuoso is invisible unless we say so."""
    c, _ = client
    driver, _stmts = _stub_driver(_edge())
    r = _decide(c, driver, "approve", AsyncMock(return_value=None))
    assert r.status_code == 200
    assert r.json()["projected"] is False


# --------------------------------------------------------------- decline


def test_decline_is_terminal_and_publishes_nothing(client):
    c, _ = client
    driver, stmts = _stub_driver(_edge())
    emit = AsyncMock()
    r = _decide(c, driver, "decline", emit)

    assert r.status_code == 200
    written = " ".join(stmts)
    assert "'declined'" in written
    # The candidate is kept, not deleted: the rules are deterministic and
    # the sweeper re-runs them forever, so a deleted decline comes back.
    assert "DELETE r" not in written
    # Declining never asserted anything, so there is nothing to retract.
    assert "NOT_SAME_AS" not in written
    emit.assert_not_awaited()


def test_keep_as_related_declines_the_equivalence(client):
    c, _ = client
    driver, stmts = _stub_driver(_edge())
    emit = AsyncMock()
    r = _decide(c, driver, "keep_as_related", emit)

    assert r.status_code == 200
    written = " ".join(stmts)
    assert "RELATED_TO" in written
    assert "'declined'" in written
    emit.assert_not_awaited()


def test_already_declined_candidate_is_not_redecidable(client):
    c, _ = client
    driver, _stmts = _stub_driver(_edge(status="declined"))
    r = _decide(c, driver, "approve", AsyncMock())
    assert r.status_code == 409


# ------------------------------------------------------------ correction


def test_correction_retracts_the_published_assertion(client):
    c, _ = client
    driver, stmts = _stub_driver(_edge())
    retract = AsyncMock(return_value=99)
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)), \
         patch("src.api.routes.candidates.eventlog.emit_retract_same_as", retract):
        r = c.post("/same-as/A/B/correct",
                   json={"reason": "different registration numbers",
                         "reviewer": "someone@fontem.eu"})

    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "manual_correction"
    assert body["retracted"] is True

    written = " ".join(stmts)
    assert "DELETE s" in written, "the wrong :SAME_AS must go"
    assert "NOT_SAME_AS" in written, "the correction must be durable"

    retract.assert_awaited_once()
    kwargs = retract.await_args.kwargs
    assert kwargs["reason"] == "different registration numbers"
    assert kwargs["retracted_method"] == "fuzzy_name_same_country"


def test_failed_retraction_is_reported(client):
    """Neo4j is corrected but the triple may still stand in Virtuoso.
    Reporting success there would be a lie with a wrong owl:sameAs
    still published."""
    c, _ = client
    driver, _stmts = _stub_driver(_edge())
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)), \
         patch("src.api.routes.candidates.eventlog.emit_retract_same_as",
               AsyncMock(return_value=None)):
        r = c.post("/same-as/A/B/correct",
                   json={"reason": "wrong", "reviewer": "x@fontem.eu"})
    assert r.status_code == 200
    assert r.json()["retracted"] is False


def test_correcting_a_pair_with_no_assertion_is_a_404(client):
    """Correction is for published claims. A proposal that was never
    approved is declined, not corrected."""
    c, _ = client
    driver, _stmts = _stub_driver(None)
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)):
        r = c.post("/same-as/A/B/correct",
                   json={"reason": "wrong", "reviewer": "x@fontem.eu"})
    assert r.status_code == 404
