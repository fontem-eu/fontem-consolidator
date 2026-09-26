"""migrations.apply: startup ensures indexes, and does nothing else.

It used to run a name_clean backfill and a SAME_AS self-loop sweep on every
boot: full label scans that grow with the graph. On a cold page cache they
outlasted the server's transaction timeout (the prod sweeper crash-looped on
2026-09-25) and then the API's liveness window (shared, 2026-09-26). Both
root causes were fixed at the source, the sink writes name_clean on every
upsert and no rule may propose a node as its own duplicate, and neither graph
held a single row either sweep would touch. A statement that scans the graph
belongs in a one-off job, not in the path a pod must finish before it serves.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.neo4j import migrations


def _recording_driver(ran: list[str]):
    session = AsyncMock()

    async def run(stmt, **_params):
        ran.append(stmt)

    session.run = run
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return driver


@pytest.mark.asyncio
async def test_startup_runs_index_statements_and_nothing_else():
    ran: list[str] = []
    with patch.object(migrations, "_drop_stale_vector_index", AsyncMock()):
        await migrations.apply(_recording_driver(ran), "neo4j")
    assert ran == list(migrations.INDEX_CYPHER)


def test_no_startup_statement_scans_or_writes_data():
    """Schema only: an index statement neither matches nor writes nodes."""
    for stmt in migrations.INDEX_CYPHER:
        assert stmt.lstrip().upper().startswith("CREATE"), stmt
        for verb in ("MATCH", " SET ", "DELETE", "MERGE"):
            assert verb not in stmt.upper(), stmt
