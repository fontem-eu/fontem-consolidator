"""Ephemeral GDS graph projection helpers.

GDS algorithms require a named in-memory graph. We project, run, and drop
each time to keep the projection cheap and ensure freshness. Projections
are namespaced with a run-scoped token so parallel consolidations don't
clobber each other.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from neo4j import AsyncDriver


def _token() -> str:
    return uuid.uuid4().hex[:12]


@asynccontextmanager
async def projected_subgraph(
    driver: AsyncDriver,
    *,
    node_query: str,
    relationship_query: str,
    database: str,
    graph_name: str | None = None,
):
    """Project a named GDS graph via Cypher projection, yield the name, drop after use."""
    name = graph_name or f"consolidator_{_token()}"
    async with driver.session(database=database) as session:
        await session.run(
            """
            CALL gds.graph.project.cypher($name, $node_q, $rel_q, {validateRelationships: false})
            """,
            name=name,
            node_q=node_query,
            rel_q=relationship_query,
        )
    try:
        yield name
    finally:
        async with driver.session(database=database) as session:
            await session.run("CALL gds.graph.drop($name, false)", name=name)


async def gds_available(driver: AsyncDriver, database: str) -> bool:
    """Probe whether the graph-data-science plugin is installed."""
    try:
        async with driver.session(database=database) as session:
            result = await session.run("CALL gds.version()")
            record = await result.single()
            return record is not None
    # Any driver/session/server error means GDS isn't usable — log nothing
    # and return False so callers fall through to the no-GDS branch.
    except Exception:  # pylint: disable=broad-exception-caught
        return False
