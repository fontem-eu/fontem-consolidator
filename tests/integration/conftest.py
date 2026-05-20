"""Integration-test fixtures.

Spins a Neo4j testcontainer with APOC + GDS preinstalled, runs the migrations,
and yields the live AsyncDriver plus a DB-reset helper.

Skip automatically if Docker is not available.
"""
# protected-access: the fixture intentionally resets and re-seeds
# `src.consolidator.neo4j.client._driver` + `rules.loader._loaded`
# + `rules.registry._REGISTRY` between tests so module-level
# caches don't leak across cases.
# import-outside-toplevel: testcontainers / docker / app modules
# are imported lazily inside fixtures so missing-dep skip paths
# and per-test patches activate before module side effects.
# redefined-outer-name: `neo4j_container` is a session fixture
# consumed by the per-test `driver` fixture — pytest's intended
# pattern.
# broad-exception-caught: docker probe + bolt-readiness wait
# legitimately swallow any startup error and back off.
# pylint: disable=protected-access,import-outside-toplevel,redefined-outer-name,broad-exception-caught

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from neo4j import AsyncDriver, AsyncGraphDatabase

# Disable testcontainers' "ryuk" resource-reaper — it fails under DinD and we
# clean up the container explicitly in the session-scoped fixture.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")


def _docker_available() -> bool:
    try:
        import docker  # noqa: F401
        import testcontainers.neo4j  # noqa: F401  pylint: disable=unused-import

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker / testcontainers unavailable — integration tests skipped.",
)


@pytest.fixture(scope="session")
def neo4j_container():
    from testcontainers.neo4j import Neo4jContainer

    container = (
        Neo4jContainer("neo4j:5.26-enterprise")
        .with_env("NEO4J_ACCEPT_LICENSE_AGREEMENT", "yes")
        .with_env("NEO4J_PLUGINS", '["apoc", "graph-data-science"]')
        .with_env("NEO4J_apoc_trigger_enabled", "true")
        .with_env(
            "NEO4J_dbms_security_procedures_unrestricted",
            "gds.*,apoc.*",
        )
        .with_env("NEO4J_dbms_security_procedures_allowlist", "gds.*,apoc.*")
    )
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest_asyncio.fixture
async def driver(neo4j_container) -> AsyncIterator[AsyncDriver]:
    uri = neo4j_container.get_connection_url()
    drv = AsyncGraphDatabase.driver(
        uri, auth=(neo4j_container.username, neo4j_container.password)
    )

    # Wait for bolt ready
    for _ in range(40):
        try:
            async with drv.session() as s:
                await s.run("RETURN 1")
            break
        except Exception:
            await asyncio.sleep(1)

    # Clean slate
    async with drv.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # Ensure common constraints ETL/consolidator rely on
        await s.run(
            "CREATE CONSTRAINT company_gmr_id IF NOT EXISTS "
            "FOR (c:Company) REQUIRE c.gmr_id IS UNIQUE"
        )
        await s.run(
            "CREATE CONSTRAINT authority_id IF NOT EXISTS "
            "FOR (a:Authority) REQUIRE a.authority_id IS UNIQUE"
        )
        # Full-text index on Company.name (fuzzy rule needs it)
        await s.run(
            "CREATE FULLTEXT INDEX company_name_ft IF NOT EXISTS "
            "FOR (c:Company) ON EACH [c.name]"
        )
        # Consolidator's own audit indexes
        from src.consolidator.neo4j.migrations import apply

        await apply(drv, "neo4j")

    # Make rules' internal get_driver() return this same test driver.
    from src.consolidator.neo4j import client as _client_module

    _client_module._driver = drv  # type: ignore[attr-defined]

    yield drv

    _client_module._driver = None  # type: ignore[attr-defined]
    await drv.close()


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure the rule registry is populated for every test module."""
    from src.consolidator.rules import registry
    from src.consolidator.rules.loader import load_all

    registry._REGISTRY.clear()  # type: ignore[attr-defined]
    # re-enable loader
    import src.consolidator.rules.loader as loader_mod

    loader_mod._loaded = False
    load_all()
    yield
