"""Tests for /relationships review endpoint.

The /candidates endpoint reviews `:SAME_AS` (entity dedup). This
endpoint reviews the resolver-driven `:REPRESENTS` and `:SANCTIONED`
edges that ETLs write with `reviewed=false`. Different action
vocabulary (accept / reject), different node-id surface (Lobbyist's
tr_id, SanctionedEntity's entity_id).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(name="client")
def _client():
    return TestClient(app, raise_server_exceptions=False)


def _record(d: dict) -> MagicMock:
    """Make a Neo4j Record-like mock from a dict."""
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: d[k]
    rec.get = d.get
    return rec


def _async_session(rows: list[dict] | None = None, single: dict | None = None):
    """Build a mock session whose `.run().__aiter__` yields rows and
    `.run().single()` returns the optional single row."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    async def _run(_query, **_kwargs):
        result = MagicMock()
        rows_list = [_record(r) for r in (rows or [])]

        async def _aiter():
            for r in rows_list:
                yield r

        result.__aiter__ = lambda self: _aiter()

        async def _single():
            return _record(single) if single else None

        result.single = _single
        return result

    session.run = AsyncMock(side_effect=_run)
    return session


def _driver(session):
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver


def test_list_rejects_unsupported_rel_type(client):
    r = client.get("/relationships?rel_type=AWARDED_TO")
    assert r.status_code == 422  # Literal validator rejects


def test_list_returns_represents_review_rows(client):
    """A REPRESENTS edge with reviewed=false should surface with both
    sides' canonical id and the edge's tier/confidence."""
    rows = [{
        "rel_type": "REPRESENTS",
        "edge_id": "edge-1",
        "a_labels": ["Lobbyist"],
        "a_props": {
            "tr_id": "888-99",
            "name": "Some NGO",
            "country_iso": "DEU",
        },
        "b_labels": ["Company"],
        "b_props": {
            "gmr_id": "gmr-c",
            "name": "Some Company",
            "country": "DEU",
        },
        "confidence": 0.95,
        "tier": "name_country",
        "method": "resolver",
        "detected_at": None,
    }]
    session = _async_session(rows=rows)
    with patch("src.api.routes.relationships.get_driver", return_value=_driver(session)):
        r = client.get("/relationships?rel_type=REPRESENTS&reviewed=false")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    item = body[0]
    assert item["rel_type"] == "REPRESENTS"
    assert item["edge_id"] == "edge-1"
    assert item["source"]["id"] == "888-99"
    assert item["target"]["id"] == "gmr-c"
    assert item["tier"] == "name_country"


def test_decide_accept_marks_edge_reviewed(client):
    edge_payload = {
        "rel_type": "REPRESENTS",
        "a_label": "Lobbyist",
        "a_props": {"tr_id": "888-99", "name": "Some NGO"},
        "b_label": "Company",
        "b_props": {"gmr_id": "gmr-c", "name": "Some Company"},
        "confidence": 0.95,
        "tier": "name_country",
        "method": "resolver",
    }
    session = _async_session(single=edge_payload)
    with patch("src.api.routes.relationships.get_driver", return_value=_driver(session)):
        r = client.post(
            "/relationships/edge-1/decide",
            json={"decision": "accept", "reviewer": "alice"},
        )
    assert r.status_code == 200
    assert r.json()["outcome"] == "manual_accept_relationship"
    # Verify a SET cypher was issued — at least one of the run calls
    # should contain "SET r.reviewed = true".
    cypher_strs = [
        call.args[0] for call in session.run.call_args_list if call.args
    ]
    assert any("SET" in q and "reviewed   = true" in q for q in cypher_strs), (
        f"expected an accept-update cypher in: {cypher_strs}"
    )


def test_decide_reject_deletes_edge(client):
    edge_payload = {
        "rel_type": "REPRESENTS",
        "a_label": "Lobbyist",
        "a_props": {"tr_id": "888-99"},
        "b_label": "Company",
        "b_props": {"gmr_id": "gmr-c"},
        "confidence": 0.95, "tier": "name_country", "method": "resolver",
    }
    session = _async_session(single=edge_payload)
    with patch("src.api.routes.relationships.get_driver", return_value=_driver(session)):
        r = client.post(
            "/relationships/edge-1/decide",
            json={"decision": "reject", "reviewer": "alice"},
        )
    assert r.status_code == 200
    assert r.json()["outcome"] == "manual_reject_relationship"
    cypher_strs = [
        call.args[0] for call in session.run.call_args_list if call.args
    ]
    assert any("DELETE r" in q for q in cypher_strs)


def test_decide_404_when_edge_missing(client):
    session = _async_session(single=None)
    with patch("src.api.routes.relationships.get_driver", return_value=_driver(session)):
        r = client.post(
            "/relationships/edge-x/decide",
            json={"decision": "accept", "reviewer": "alice"},
        )
    assert r.status_code == 404


def test_decide_rejects_same_as_edges(client):
    """SAME_AS edges have their own /candidates flow — refuse to act
    on them here so callers don't bypass the merge/reject/keep_as_related
    semantics that endpoint enforces."""
    edge_payload = {
        "rel_type": "SAME_AS",
        "a_label": "Company", "a_props": {"gmr_id": "x"},
        "b_label": "Company", "b_props": {"gmr_id": "y"},
        "confidence": 0.95, "tier": None, "method": "fuzzy",
    }
    session = _async_session(single=edge_payload)
    with patch("src.api.routes.relationships.get_driver", return_value=_driver(session)):
        r = client.post(
            "/relationships/edge-x/decide",
            json={"decision": "accept", "reviewer": "alice"},
        )
    assert r.status_code == 400
    assert "SAME_AS" in r.text or "/candidates" in r.text
