"""Unit tests for /resolve/batch.

The batch endpoint addresses the 'one HTTP per ETL row' overhead
that started biting once the lobbying loader (16k+ rows) and TED
matcher (per-awardee) migrated to /resolve.

Behavioural contract:
- Output length == input length, in the same order.
- Empty rows (no usable attributes) get a `no_match` slot rather
  than being dropped silently — keeps positional alignment.
- Batches over the cap return 400.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(name="client")
def _client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(name="patched_resolver")
def _patched_resolver():
    """Stub `resolver.resolve` so unit tests don't need a Neo4j."""
    from src.consolidator.resolver import (  # pylint: disable=import-outside-toplevel
        ResolveMatch, ResolveResult,
    )

    async def _fake(*_args, name=None, **_kwargs):
        # Return a tier-1 LEI hit for any row whose name starts with "Match",
        # otherwise no_match. This is enough to differentiate the slots.
        if name and name.startswith("Match"):
            return ResolveResult(
                hint="matched",
                match=ResolveMatch(
                    gmr_id=f"gmr-{name}",
                    name=name, country="DEU", lei="X" * 20,
                    tier="lei", confidence=1.0,
                ),
                normalised_country="DEU",
            )
        return ResolveResult(hint="no_match", normalised_country="DEU")

    with patch("src.consolidator.resolver.resolve", side_effect=_fake), \
         patch("src.api.routes.resolve.get_driver", new=AsyncMock(return_value=None)):
        yield


def test_batch_preserves_order_and_length(client, patched_resolver):  # pylint: disable=unused-argument
    rows = [
        {"name": "Match One Inc", "country": "DE"},
        {"name": "Other Inc", "country": "DE"},
        {"name": "Match Two Inc", "country": "DE"},
    ]
    r = client.post(
        "/resolve/batch",
        json={"entity_type": "Company", "rows": rows},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert body["results"][0]["hint"] == "matched"
    assert body["results"][0]["match"]["gmr_id"] == "gmr-Match One Inc"
    assert body["results"][1]["hint"] == "no_match"
    assert body["results"][2]["hint"] == "matched"
    assert body["results"][2]["match"]["gmr_id"] == "gmr-Match Two Inc"


def test_batch_empty_row_yields_no_match_slot(client, patched_resolver):  # pylint: disable=unused-argument
    """Rows with no attributes whatsoever are surfaced as no_match
    rather than dropped — callers MUST be able to align by index."""
    r = client.post(
        "/resolve/batch",
        json={
            "entity_type": "Company",
            "rows": [
                {"name": "Match Real Inc", "country": "DE"},
                {},  # empty
                {"name": "Match Other Inc", "country": "DE"},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert body["results"][1]["hint"] == "no_match"


def test_batch_size_cap(client):
    """Batches over the cap are rejected — protects the consolidator
    from an HTTP request that could pin the worker for minutes."""
    rows = [{"name": f"Org {i:04d}", "country": "DE"} for i in range(201)]
    r = client.post(
        "/resolve/batch",
        json={"entity_type": "Company", "rows": rows},
    )
    assert r.status_code == 400
    assert "exceeds limit" in r.text.lower() or "limit" in r.text.lower()


def test_batch_empty_returns_empty(client):
    r = client.post(
        "/resolve/batch",
        json={"entity_type": "Company", "rows": []},
    )
    assert r.status_code == 200
    assert r.json() == {"results": []}


def test_batch_entity_type_required(client):
    r = client.post(
        "/resolve/batch",
        json={"rows": [{"name": "X", "country": "DE"}]},
    )
    assert r.status_code == 422
