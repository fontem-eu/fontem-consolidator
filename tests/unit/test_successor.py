"""Unit tests for SuccessorLeiMatch — Neo4j mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.company.successor import SuccessorLeiMatch


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
async def test_applies_only_when_entity_is_active_with_lei():
    rule = SuccessorLeiMatch()
    assert await rule.applies(Entity("Company", "A", {"lei": "X", "name": "n", "country": "FR", "active": True})) is True
    # inactive self → don't initiate
    assert await rule.applies(Entity("Company", "A", {"lei": "X", "name": "n", "country": "FR", "active": False})) is False
    # missing lei
    assert await rule.applies(Entity("Company", "A", {"name": "n", "country": "FR", "active": True})) is False
    # missing name
    assert await rule.applies(Entity("Company", "A", {"lei": "X", "country": "FR", "active": True})) is False


@pytest.mark.asyncio
async def test_resolve_carries_retired_lei_in_details():
    rule = SuccessorLeiMatch()
    entity = Entity("Company", "A", {"lei": "529900ACTIVEXXXXXXX1", "active": True})
    candidate = Candidate(
        entity=Entity("Company", "B", {"lei": "529900RETIREDXXXXXX2", "active": False}),
        context={"retired_lei": "529900RETIREDXXXXXX2"},
    )
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "merge"
    assert decision.rule_name == "successor_lei_match"
    assert decision.confidence == 0.98
    assert decision.details["retired_lei"] == "529900RETIREDXXXXXX2"


@pytest.mark.asyncio
async def test_find_candidates_contract_shape():
    """Confirm the rule packages the retired LEI into candidate.context so
    resolve() can propagate it to the action executor."""
    rule = SuccessorLeiMatch()
    entity = Entity(
        "Company", "A",
        {"lei": "529900ACTIVEXXXXXXX1", "name": "kleiner und bold GmbH",
         "country": "DEU", "active": True},
    )
    rec = {"b": {"gmr_id": "B", "lei": "529900RETIREDXXXXXX2",
                 "name": "kleiner und bold GmbH", "country": "DEU", "active": False}}
    driver = _fake_driver([rec])
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock(return_value=driver)):
        candidates = await rule.find_candidates(entity)
    assert len(candidates) == 1
    assert candidates[0].entity.id == "B"
    assert candidates[0].context["retired_lei"] == "529900RETIREDXXXXXX2"


@pytest.mark.asyncio
async def test_rule_registered_in_loader():
    """Pipeline ordering: successor runs after exact-id rules but before
    name-country so it picks up the retire-pair case before the exact-name
    rule downgrades it to a conflict."""
    from src.consolidator.rules.loader import load_all
    from src.consolidator.rules.registry import list_rules, _REGISTRY  # noqa

    _REGISTRY.clear()
    import src.consolidator.rules.loader as L
    L._loaded = False
    load_all()
    names = [r.name for r in list_rules()]
    assert "successor_lei_match" in names
    # Runs before exact_name_country_match (higher confidence)
    assert names.index("successor_lei_match") < names.index("exact_name_country_match")
