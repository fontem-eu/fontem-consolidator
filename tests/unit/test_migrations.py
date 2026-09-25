"""migrations.apply: index creation is fatal, the backfill is not."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import ClientError, Neo4jError

from src.consolidator.neo4j import migrations


def _driver_raising(exc):
    session = AsyncMock()
    session.run = AsyncMock(side_effect=exc)
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return driver


def _error(code: str) -> Neo4jError:
    # Same construction as test_migrations_race: ClientError.code is a
    # read-only property, so the error has to be hydrated, not patched.
    return Neo4jError._hydrate_neo4j(  # pylint: disable=protected-access
        code=code, message="raised by the fake driver",
    )

@pytest.mark.asyncio
async def test_a_backfill_timeout_does_not_kill_the_process():
    """The prod sweeper crash-looped on 2026-09-25 (9 restarts): each
    boot it consolidated hundreds of entities, then migrations.apply's
    name_clean backfill — a full label scan, run every boot, against a
    Neo4j whose page cache was cold after a cluster restart — blew the
    server transaction timeout and took the process with it.

    name_clean is best-effort here (the sink writes it on every upsert),
    so a timeout must degrade, not kill."""
    driver = _driver_raising(_error(
        "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration"))

    with patch.object(migrations, "_ensure", AsyncMock()), \
         patch.object(migrations, "_drop_stale_vector_index", AsyncMock()):
        await migrations.apply(driver, "neo4j")   # must not raise


@pytest.mark.asyncio
async def test_a_real_backfill_error_still_raises():
    """Only 'the graph was busy' is skipped; a broken statement must
    still be loud."""
    driver = _driver_raising(_error("Neo.ClientError.Statement.SyntaxError"))

    with patch.object(migrations, "_ensure", AsyncMock()), \
         patch.object(migrations, "_drop_stale_vector_index", AsyncMock()), \
         pytest.raises(ClientError):
        await migrations.apply(driver, "neo4j")
