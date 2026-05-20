from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.rules.authority.basic import (
    ExactAuthorityIdMatch,
    ExactNameCountryMatchAuthority,
)
from src.consolidator.rules.base import Entity


def _fake_driver(records):
    session = AsyncMock()

    class _Result:
        def __init__(self, recs):
            self._recs = recs
            self._it = iter(recs)

        def __aiter__(self):
            self._it = iter(self._recs)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    session.run = AsyncMock(return_value=_Result(records))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    driver = AsyncMock()
    driver.session = MagicMock(return_value=ctx)
    return driver


async def _find(rule, entity, records):
    driver = _fake_driver(records)
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock(return_value=driver)):
        return await rule.find_candidates(entity)


@pytest.mark.asyncio
async def test_authority_id_match_merges():
    rule = ExactAuthorityIdMatch()
    entity = Entity("Authority", "auth-1", {"authority_id": "auth-1", "name": "Min Fin"})
    rec = {"a": {"authority_id": "auth-2", "name": "Min Fin"}}
    cands = await _find(rule, entity, [rec])
    assert len(cands) == 1
    assert (await rule.resolve(entity, cands[0])).action == "merge"


@pytest.mark.asyncio
async def test_authority_name_country_merges():
    rule = ExactNameCountryMatchAuthority()
    entity = Entity(
        "Authority",
        "auth-1",
        {"authority_id": "auth-1", "name": "Ministry of Finance", "country": "FR"},
    )
    rec = {
        "a": {
            "authority_id": "auth-2",
            "name": "Ministry of Finance",
            "country": "FR",
        }
    }
    cands = await _find(rule, entity, [rec])
    assert len(cands) == 1
    assert (await rule.resolve(entity, cands[0])).action == "merge"
