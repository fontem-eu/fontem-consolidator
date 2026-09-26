"""Losing the index-creation race is not a startup failure.

CREATE ... IF NOT EXISTS checks for the rule and then creates it, not
atomically. Both consolidator replicas and the sweeper run
migrations.apply() at startup, so when a node restart brings them up
together one of them can find the index created between its check and
its create. On prod 2026-09-10 the sweeper exited 1 on company_cik and
came back on the next attempt; after every node restart it was a coin
toss whether the pod's first start would count.
"""
import pytest
from neo4j.exceptions import ClientError, Neo4jError

from src.consolidator.neo4j import migrations


def _error(code: str) -> Neo4jError:
    return Neo4jError._hydrate_neo4j(  # pylint: disable=protected-access
        code=code, message="raised by the fake driver",
    )


class _Result:
    async def single(self):
        return None


class _Session:
    def __init__(self, driver: "_Driver"):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def run(self, stmt, **_params):
        self._driver.ran.append(stmt)
        if stmt in self._driver.fail:
            raise self._driver.fail[stmt]
        return _Result()


class _Driver:
    def __init__(self, fail: dict | None = None):
        self.fail = fail or {}
        self.ran: list[str] = []

    def session(self, database=None):  # noqa: D401 - mimics AsyncDriver
        assert database == "neo4j"
        return _Session(self)


def _company_cik() -> str:
    return next(s for s in migrations.INDEX_CYPHER if "company_cik" in s)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", sorted(migrations._CREATED_CONCURRENTLY))  # pylint: disable=protected-access
async def test_an_index_created_concurrently_is_success(code):
    driver = _Driver({_company_cik(): _error(code)})
    await migrations.apply(driver, "neo4j")
    # Every statement after the one that lost the race still ran.
    assert set(migrations.INDEX_CYPHER) <= set(driver.ran)
    assert driver.ran[-1] == migrations.INDEX_CYPHER[-1]


@pytest.mark.asyncio
async def test_any_other_schema_error_still_fails_startup():
    driver = _Driver({_company_cik(): _error("Neo.ClientError.Statement.SyntaxError")})
    with pytest.raises(ClientError):
        await migrations.apply(driver, "neo4j")
