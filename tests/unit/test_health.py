from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch(
        "src.consolidator.neo4j.client.AsyncGraphDatabase.driver"
    ) as mock_driver, patch(
        "src.consolidator.neo4j.migrations.apply", new=AsyncMock()
    ):
        mock_driver.return_value = AsyncMock()
        from src.api.app import app

        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_rules_endpoint_returns_list(client):
    r = client.get("/rules")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
