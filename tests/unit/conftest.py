"""Shared fixture: TestClient with Neo4j driver fully stubbed.

Uses a patched `get_driver` / `close_driver` pair so the app lifespan doesn't
touch real I/O at startup or shutdown and doesn't leak a real MagicMock
through the module-level `_driver` global.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.consolidator.neo4j import client as neo4j_client

    fake_driver = AsyncMock()

    async def _get():
        return fake_driver

    async def _close():
        pass

    # Reset any leftover global state between tests
    neo4j_client._driver = None  # type: ignore[attr-defined]

    with patch("src.consolidator.neo4j.client.get_driver", _get), patch(
        "src.consolidator.neo4j.client.close_driver", _close
    ), patch(
        "src.api.app.neo4j_client.get_driver", _get
    ), patch(
        "src.api.app.neo4j_client.close_driver", _close
    ), patch(
        "src.consolidator.neo4j.migrations.apply", new=AsyncMock()
    ):
        from src.api.app import app

        with TestClient(app) as c:
            yield c, fake_driver
