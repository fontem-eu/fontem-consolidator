"""Tests for POST /events/dispatch.

Covers:
  - Upsert event types route to the right (entity_type, entity_id) call.
  - Filing events consolidate the *parent* Company.
  - Delete and unmapped event types 200-noop.
  - Missing payload key → 400.
"""
from unittest.mock import AsyncMock, patch


def _payload(event_type: str, payload: dict, *, seq: int = 1, iri: str = "x") -> dict:
    return {
        "seq": seq,
        "event_type": event_type,
        "iri": iri,
        "domain": "test",
        "payload": payload,
        "batch_id": None,
        "producer": "test",
    }


def test_dispatch_upsert_company_routes_to_company(client):
    c, _driver = client
    fake_result = type("R", (), {
        "run_id": "rid",
        "entity_type": "Company",
        "entity_id": "abc",
        "decisions": [],
        "rules_fired": 0,
    })()
    with patch(
        "src.api.routes.dispatch.engine.consolidate",
        new=AsyncMock(return_value=fake_result),
    ) as mock_cons:
        r = c.post(
            "/events/dispatch",
            json=_payload("UpsertCompany", {"gmr_id": "abc"}),
        )
    assert r.status_code == 200
    assert r.json()["outcome"] == "consolidated"
    mock_cons.assert_awaited_once()
    kwargs = mock_cons.await_args.kwargs
    assert kwargs["entity_type"] == "Company"
    assert kwargs["entity_id"] == "abc"
    assert kwargs["triggered_by"] == "event:1"


def test_dispatch_upsert_filing_consolidates_parent_company(client):
    """Filings carry the parent's gmr_id; the consolidator runs on the
    Company, not on the filing itself."""
    c, _driver = client
    fake_result = type("R", (), {
        "run_id": "rid", "entity_type": "Company", "entity_id": "company-99",
        "decisions": [], "rules_fired": 0,
    })()
    with patch(
        "src.api.routes.dispatch.engine.consolidate",
        new=AsyncMock(return_value=fake_result),
    ) as mock_cons:
        r = c.post(
            "/events/dispatch",
            json=_payload(
                "UpsertFiling",
                {"gmr_id": "company-99", "year": 2025, "source": "edgar"},
            ),
        )
    assert r.status_code == 200
    kwargs = mock_cons.await_args.kwargs
    assert kwargs["entity_type"] == "Company"
    assert kwargs["entity_id"] == "company-99"


def test_dispatch_delete_company_is_noop(client):
    c, _driver = client
    with patch(
        "src.api.routes.dispatch.engine.consolidate",
        new=AsyncMock(),
    ) as mock_cons:
        r = c.post(
            "/events/dispatch",
            json=_payload("DeleteCompany", {"gmr_id": "abc"}),
        )
    assert r.status_code == 200
    assert r.json()["outcome"] == "noop"
    mock_cons.assert_not_awaited()


def test_dispatch_unmapped_event_type_is_noop(client):
    """Trigger filters by INPUT_TYPES so this shouldn't normally happen,
    but if a stray AssertSameAs reaches the endpoint we 200-noop instead
    of 5xx — defence in depth against the cycle."""
    c, _driver = client
    with patch(
        "src.api.routes.dispatch.engine.consolidate",
        new=AsyncMock(),
    ) as mock_cons:
        r = c.post(
            "/events/dispatch",
            json=_payload("AssertSameAs", {"a_iri": "x", "b_iri": "y"}),
        )
    assert r.status_code == 200
    assert r.json()["outcome"] == "noop"
    mock_cons.assert_not_awaited()


def test_dispatch_missing_payload_key_400s(client):
    c, _driver = client
    with patch(
        "src.api.routes.dispatch.engine.consolidate",
        new=AsyncMock(),
    ) as mock_cons:
        r = c.post(
            "/events/dispatch",
            json=_payload("UpsertCompany", {}),  # no gmr_id
        )
    assert r.status_code == 400
    mock_cons.assert_not_awaited()


def test_dispatch_excludes_only_node_similarity_gds_rules(client):
    """Per-event dispatch must skip the expensive
    gds_node_similarity_* rules but KEEP the cheap
    gds_same_as_cluster_collapse_* rules — the latter is how
    reviewed SAME_AS clusters actually become merges. A bare
    ``gds_`` prefix would disable both, so we pin the more
    specific one."""
    c, _driver = client
    fake_result = type("R", (), {
        "run_id": "rid", "entity_type": "Company", "entity_id": "abc",
        "decisions": [], "rules_fired": 0,
    })()
    with patch(
        "src.api.routes.dispatch.engine.consolidate",
        new=AsyncMock(return_value=fake_result),
    ) as mock_cons:
        c.post(
            "/events/dispatch",
            json=_payload("UpsertCompany", {"gmr_id": "abc"}),
        )
    kwargs = mock_cons.await_args.kwargs
    assert kwargs["exclude_rule_prefix"] == "gds_node_similarity_", (
        "If this changes back to 'gds_' (or to None), reviewed-cluster "
        "collapse stops happening — duplicates accumulate after human review."
    )
