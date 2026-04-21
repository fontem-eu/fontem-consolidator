"""Unit tests for ExactNameAnyCountryAuthority — Neo4j mocked.

Targets the pattern where EU-wide bodies (EEAS, JRC, eu-LISA) appear once
per contracting-destination country as N duplicate authority nodes with
the same name but different country values.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.rules.authority.basic import ExactNameAnyCountryAuthority
from src.consolidator.rules.base import Candidate, Entity


def _fake_driver(records):
    session = AsyncMock()

    class _Result:
        def __init__(self, recs):
            self._recs = recs

        def __aiter__(self):
            self._it = iter(self._recs)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    session.run = AsyncMock(return_value=_Result(records))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    driver = AsyncMock()
    driver.session = MagicMock(return_value=ctx)
    return driver


@pytest.mark.asyncio
async def test_applies_when_entity_has_name():
    rule = ExactNameAnyCountryAuthority()
    assert await rule.applies(Entity("Authority", "A", {"name": "EEAS", "country": "FR"})) is True
    assert await rule.applies(Entity("Authority", "A", {"country": "FR"})) is False


@pytest.mark.asyncio
async def test_resolve_emits_flag_with_country_metadata():
    rule = ExactNameAnyCountryAuthority()
    entity = Entity(
        "Authority", "AID",
        {"name": "European External Action Service (EEAS)", "country": "BEL"},
    )
    candidate = Candidate(
        entity=Entity(
            "Authority", "BID",
            {"name": "European External Action Service (EEAS)", "country": "MUS"},
        ),
        context={"cross_country": True},
    )
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "flag"
    assert decision.confidence == 0.90
    assert decision.details["source_country"] == "BEL"
    assert decision.details["target_country"] == "MUS"
    assert decision.details["cross_country"] is True


@pytest.mark.asyncio
async def test_find_candidates_returns_cross_country_pairs():
    rule = ExactNameAnyCountryAuthority()
    entity = Entity(
        "Authority", "A",
        {"name": "European External Action Service (EEAS)", "country": "BEL"},
    )
    rec = {"a": {"authority_id": "B", "name": "European External Action Service (EEAS)",
                 "country": "MUS"}}
    driver = _fake_driver([rec])
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock(return_value=driver)):
        cands = await rule.find_candidates(entity)
    assert len(cands) == 1
    assert cands[0].entity.id == "B"
    assert cands[0].entity.properties["country"] == "MUS"


@pytest.mark.asyncio
async def test_rule_registered_in_loader():
    from src.consolidator.rules.loader import load_all
    from src.consolidator.rules.registry import _REGISTRY, list_rules

    _REGISTRY.clear()
    import src.consolidator.rules.loader as L
    L._loaded = False
    load_all()
    names = [r.name for r in list_rules()]
    assert "exact_name_any_country_authority" in names
    # Runs AFTER the same-country exact (0.95) so auto-merges win when applicable
    assert names.index("exact_name_country_match_authority") < names.index(
        "exact_name_any_country_authority"
    )
