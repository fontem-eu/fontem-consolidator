"""Unit tests for ExactNameAnyCountryAuthority — Neo4j mocked.

Targets the pattern where EU-wide bodies (EEAS, JRC, eu-LISA) appear once
per contracting-destination country as N duplicate authority nodes with
the same name but different country values.
"""
# protected-access: the loader's `_loaded` / registry `_REGISTRY`
# are reset between tests so the per-test rule set is deterministic.
# import-outside-toplevel: loader / registry imported inside the
# registration test so the reset happens before module-import
# side effects.
# pylint: disable=protected-access,import-outside-toplevel

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.rules.authority.basic import ExactNameAnyCountryAuthority
from src.consolidator.rules.base import Candidate, Entity


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


def _pair(name, country_a, country_b, name_b=None):
    return (
        Entity("Authority", "AID", {"name": name, "country": country_a}),
        Candidate(
            entity=Entity("Authority", "BID", {"name": name_b or name, "country": country_b}),
            context={"cross_country": True},
        ),
    )


# Pairs from the first prod sweep (2026-09-18) that ARE one body: EU
# institutions TED lists per destination country, and France with one of
# its overseas territories. These keep the confidence that auto-merges.
@pytest.mark.asyncio
@pytest.mark.parametrize("name,a,b", [
    ("European Commission, DG COMM - Communication", "CZE", "LTU"),
    ("Commission européenne, INTPA - International Partnerships", "MRT", "MDG"),
    ("European Parliament, INLO - Directorate-General for Infrastructure and Logistics",
     "BEL", "LUX"),
    ("European Union, represented by the European Commission, on behalf of and for the",
     "TJK", "BEL"),
    ("Joint Research Centre", "DEU", "ITA"),
    ("European Medicines Agency (EMA)", "NLD", "GBR"),
    ("CA Sud Basse-Terre", "GLP", "FRA"),
    ("syndicat des Hirondelles", "REU", "FRA"),
])
async def test_one_body_across_countries_keeps_auto_merge_confidence(name, a, b):
    rule = ExactNameAnyCountryAuthority()
    decision = await rule.resolve(*_pair(name, a, b))
    assert decision.action == "flag"
    assert decision.confidence >= rule.auto_merge_threshold
    assert decision.details["same_body_across_countries"] is True


# Pairs from the same sweep that are two institutions sharing a name.
# They must stay below the threshold, i.e. in the human review queue.
@pytest.mark.asyncio
@pytest.mark.parametrize("name,a,b", [
    ("Ministry of Foreign Affairs", "DNK", "NLD"),
    ("Ministère de la Défense", "BEL", "FRA"),
    ("Υπουργείο Υγείας", "CYP", "GRC"),
    ("Department of Transport", "IRL", "GBR"),
    ("Finanstilsynet", "NOR", "DNK"),
    ("INSPECTORATUL GENERAL AL POLITIEI DE FRONTIERA", "ROU", "MDA"),
    ("Ambassade Royale du Danemark", "BFA", "MLI"),
    ("UM", "FRA", "SVN"),
    ("Nice", "FRA", "GBR"),
    # Mention Europe, but national ministries with twins elsewhere.
    ("Ministère des Affaires étrangères et européennes", "LUX", "FRA"),
    ("Ministry of Foreign and European Affairs", "HRV", "SVK"),
])
async def test_shared_name_in_two_countries_goes_to_review(name, a, b):
    rule = ExactNameAnyCountryAuthority()
    decision = await rule.resolve(*_pair(name, a, b))
    assert decision.action == "flag"
    assert decision.confidence < rule.auto_merge_threshold
    assert decision.details["same_body_across_countries"] is False
    assert decision.details["source_country"] == a
    assert decision.details["target_country"] == b


@pytest.mark.asyncio
async def test_missing_country_is_not_treated_as_france():
    rule = ExactNameAnyCountryAuthority()
    decision = await rule.resolve(*_pair("Service immobilier", None, "LUX"))
    assert decision.confidence < rule.auto_merge_threshold
