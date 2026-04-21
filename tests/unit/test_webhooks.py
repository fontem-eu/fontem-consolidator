"""/webhooks/neo4j-trigger route — accepts the APOC trigger payload and dispatches async."""

from unittest.mock import AsyncMock, patch


def test_trigger_accepts_company_payload(client):
    c, _ = client
    with patch("src.api.routes.webhooks.engine.consolidate", new=AsyncMock()):
        resp = c.post(
            "/webhooks/neo4j-trigger",
            json={"label": "Company", "gmr_id": "gmr-123"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["entity_type"] == "Company"
    assert body["entity_id"] == "gmr-123"


def test_trigger_accepts_authority_payload(client):
    c, _ = client
    with patch("src.api.routes.webhooks.engine.consolidate", new=AsyncMock()):
        resp = c.post(
            "/webhooks/neo4j-trigger",
            json={"label": "Authority", "authority_id": "auth-1"},
        )
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


def test_trigger_rejects_missing_id(client):
    c, _ = client
    resp = c.post("/webhooks/neo4j-trigger", json={"label": "Company"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == "missing_id"


def test_rules_endpoint_lists_all(client):
    c, _ = client
    resp = c.get("/rules")
    assert resp.status_code == 200
    rules = resp.json()
    names = {r["name"] for r in rules}
    assert "exact_lei_match" in names
    assert "exact_vat_match" in names
    assert "exact_name_country_match" in names
    assert "fuzzy_name_same_country" in names
    assert "gds_node_similarity_company" in names
    assert "gds_node_similarity_authority" in names
    assert "gds_same_as_cluster_collapse_company" in names
    assert "exact_authority_id_match" in names
    assert len(rules) >= 12
