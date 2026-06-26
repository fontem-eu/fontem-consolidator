"""Rule unit tests — mock the Neo4j driver/session so rules can be exercised in isolation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.rules.base import Entity
from src.consolidator.rules.company.exact_identifiers import (
    ExactCikMatch,
    ExactLeiMatch,
    ExactVatMatch,
)
from src.consolidator.rules.company.name_country import ExactNameCountryMatch


def _fake_session(records):
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


async def _run_find(rule, entity, fake_records):
    driver = _fake_session(fake_records)
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock(return_value=driver)):
        return await rule.find_candidates(entity)


@pytest.mark.asyncio
async def test_exact_lei_match_skips_when_no_lei():
    rule = ExactLeiMatch()
    entity = Entity("Company", "gmr-123", {"name": "Acme"})
    assert await rule.applies(entity) is False


@pytest.mark.asyncio
async def test_exact_lei_match_emits_merge_decision():
    rule = ExactLeiMatch()
    entity = Entity("Company", "gmr-A", {"lei": "ABC", "name": "Acme"})
    record = {"c": {"gmr_id": "gmr-B", "lei": "ABC", "name": "Acme Inc"}}
    candidates = await _run_find(rule, entity, [record])
    assert len(candidates) == 1
    assert candidates[0].entity.id == "gmr-B"
    decision = await rule.resolve(entity, candidates[0])
    assert decision.action == "merge"
    assert decision.confidence == 1.0
    assert decision.target_id == "gmr-B"


@pytest.mark.asyncio
async def test_exact_vat_match_emits_merge():
    rule = ExactVatMatch()
    entity = Entity("Company", "gmr-A", {"vat": "FR123", "name": "Acme"})
    record = {"c": {"gmr_id": "gmr-B", "vat": "FR123", "name": "Acme France"}}
    candidates = await _run_find(rule, entity, [record])
    assert len(candidates) == 1
    assert (await rule.resolve(entity, candidates[0])).action == "merge"


@pytest.mark.asyncio
async def test_exact_cik_match_emits_merge():
    rule = ExactCikMatch()
    entity = Entity("Company", "gmr-A", {"cik": "0000123", "name": "Acme"})
    record = {"c": {"gmr_id": "gmr-B", "cik": "0000123"}}
    candidates = await _run_find(rule, entity, [record])
    assert (await rule.resolve(entity, candidates[0])).confidence == 1.0


LEI_VALID_A = "529900WTOG7RHO5TCH58"
LEI_VALID_B = "529900PC9XG1KHIJD788"


@pytest.mark.asyncio
async def test_name_country_match_detects_lei_conflict():
    """Same name + country but different canonical LEIs → must refuse merge, emit conflict flag."""
    rule = ExactNameCountryMatch()
    entity = Entity(
        "Company", "gmr-A", {"name": "Acme Holdings", "country": "FR", "lei": LEI_VALID_A}
    )
    candidate_rec = {
        "c": {"gmr_id": "gmr-B", "name": "Acme Holdings", "country": "FR", "lei": LEI_VALID_B}
    }
    candidates = await _run_find(rule, entity, [candidate_rec])
    assert len(candidates) == 1
    decision = await rule.resolve(entity, candidates[0])
    assert decision.action == "flag"
    assert decision.details["conflict"] is True
    assert decision.details["conflicting_property"] == "lei"


@pytest.mark.asyncio
async def test_name_country_match_flags_when_both_leis_malformed():
    """If both 'LEIs' are malformed (non-canonical), they're effectively unknown —
    no conflict, but name+country still only FLAGs for review (never auto-merges:
    ~0.27% false-merge floor on name alone)."""
    rule = ExactNameCountryMatch()
    entity = Entity("Company", "gmr-A", {"name": "Acme", "country": "FR", "lei": "LEI-A"})
    candidate_rec = {"c": {"gmr_id": "gmr-B", "name": "Acme", "country": "FR", "lei": "LEI-B"}}
    candidates = await _run_find(rule, entity, [candidate_rec])
    decision = await rule.resolve(entity, candidates[0])
    assert decision.action == "flag"
    assert not decision.details.get("conflict")


@pytest.mark.asyncio
async def test_name_country_match_flags_without_conflict():
    """A clean name+country match is a plain review flag — never an auto-merge."""
    rule = ExactNameCountryMatch()
    entity = Entity("Company", "gmr-A", {"name": "Acme Holdings", "country": "FR"})
    candidate_rec = {"c": {"gmr_id": "gmr-B", "name": "Acme Holdings", "country": "FR"}}
    candidates = await _run_find(rule, entity, [candidate_rec])
    decision = await rule.resolve(entity, candidates[0])
    assert decision.action == "flag"
    assert decision.confidence == 0.95
    assert decision.details.get("name_country_review") is True
