"""FuzzyNameSameCountry — Jaro-Winkler on normalised names, skip on conflict.

Neo4j is mocked. We assert the rule produces the expected candidates
(filtered by JW threshold) and that resolve() skips pairs whose hard
identifiers disagree canonically.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.company.fuzzy import (
    FuzzyNameSameCountry,
    _normalise,
)


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


def test_normalise_strips_legal_suffixes():
    assert _normalise("Acme Holdings SA") == "ACME HOLDINGS"
    assert _normalise("Fållen AB") == "FÅLLEN"
    assert _normalise("MacNeill Properties LLP") == "MACNEILL PROPERTIES"
    assert _normalise("Deutsche Bank AG") == "DEUTSCHE BANK"


def test_normalise_handles_punctuation():
    assert _normalise("ACME, S.A.R.L.") == "ACME"
    assert _normalise("kleiner & bold GmbH") == "KLEINER BOLD"


def test_normalise_strips_ukrainian_llc_boilerplate():
    # Ukrainian LLC prefix — what matters is inside the quotes
    assert _normalise('ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ "АНСУ"').strip().strip('"') == 'АНСУ'
    # Double space between words — still collapses
    assert _normalise('ТОВАРИСТВО З ОБМЕЖЕНОЮ  ВІДПОВІДАЛЬНІСТЮ "ГЕТЬМАНСЬКЕ"').strip().strip('"') == 'ГЕТЬМАНСЬКЕ'


def test_normalise_strips_short_ukrainian_tov():
    assert _normalise('ТОВ "Егзагон"').strip().strip('"') == 'ЕГЗАГОН'


def test_normalise_strips_polish_long_form():
    assert _normalise('PK SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ').strip() == 'PK'
    assert _normalise('Skamex Spółka Akcyjna').strip() == 'SKAMEX'


def test_normalise_strips_baltic_forms():
    assert _normalise('UAB "Gama Projects"').strip().strip('"') == 'GAMA PROJECTS'
    assert _normalise('SIA Megacom').strip() == 'MEGACOM'


def test_normalise_strips_czech_slovak():
    assert _normalise('Moravia s.r.o.').strip() == 'MORAVIA'
    assert _normalise('ACME a.s.').strip() == 'ACME'


def test_normalise_kiloutou_variants_collapse():
    # The practical regression: KILOUTOU / SAS KILOUTOU / KILOUTOU SAS all normalise identically
    a = _normalise('KILOUTOU').strip()
    b = _normalise('SAS KILOUTOU').strip()
    c = _normalise('KILOUTOU SAS').strip()
    assert a == b == c == 'KILOUTOU'


@pytest.mark.asyncio
async def test_fuzzy_rejects_parent_subsidiary():
    """'SOCOTEC' vs 'SOCOTEC CONSTRUCTION' must NOT emit a flag (Jaro-Winkler < 0.92)."""
    rule = FuzzyNameSameCountry()
    entity = Entity("Company", "gmr-A", {"name": "SOCOTEC", "country": "FRA"})
    rec = {"node": {"gmr_id": "gmr-B", "name": "SOCOTEC CONSTRUCTION", "country": "FRA"}, "score": 5.0}
    driver = _fake_driver([rec])
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock(return_value=driver)):
        out = await rule.find_candidates(entity)
    assert out == []


@pytest.mark.asyncio
async def test_fuzzy_rejects_shared_legal_suffix_noise():
    """'Fållen AB' vs 'AB Ility AB' should not match — both collapse to mostly noise."""
    rule = FuzzyNameSameCountry()
    entity = Entity("Company", "gmr-A", {"name": "Fållen AB", "country": "SWE"})
    rec = {"node": {"gmr_id": "gmr-B", "name": "AB Ility AB", "country": "SWE"}, "score": 3.0}
    driver = _fake_driver([rec])
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock(return_value=driver)):
        out = await rule.find_candidates(entity)
    assert out == []


@pytest.mark.asyncio
async def test_fuzzy_accepts_near_identical_names():
    """'Acme Holdings SA' vs 'ACME Holdings S.A.' normalises to same tokens → 1.0."""
    rule = FuzzyNameSameCountry()
    entity = Entity("Company", "gmr-A", {"name": "Acme Holdings SA", "country": "FRA"})
    rec = {"node": {"gmr_id": "gmr-B", "name": "ACME Holdings S.A.", "country": "FRA"}, "score": 8.0}
    driver = _fake_driver([rec])
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock(return_value=driver)):
        out = await rule.find_candidates(entity)
    assert len(out) == 1
    assert out[0].context["jw_similarity"] >= 0.92


@pytest.mark.asyncio
async def test_fuzzy_resolve_skips_when_hard_ids_conflict():
    """Even with similar names, different canonical LEIs → noop, not flag."""
    rule = FuzzyNameSameCountry()
    entity = Entity(
        "Company",
        "gmr-A",
        {"name": "Acme Holdings", "country": "FR", "lei": "529900WTOG7RHO5TCH58"},
    )
    candidate = Candidate(
        entity=Entity(
            "Company",
            "gmr-B",
            {"name": "ACME Holdings", "country": "FR", "lei": "529900PC9XG1KHIJD788"},
        ),
        context={"jw_similarity": 0.99},
    )
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "noop"
    assert decision.details["skipped"] == "hard_id_conflict"


@pytest.mark.asyncio
async def test_fuzzy_resolve_flags_when_ids_do_not_conflict():
    rule = FuzzyNameSameCountry()
    entity = Entity("Company", "gmr-A", {"name": "Acme Holdings", "country": "FR"})
    candidate = Candidate(
        entity=Entity("Company", "gmr-B", {"name": "ACME Holdings", "country": "FR"}),
        context={"jw_similarity": 0.97},
    )
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "flag"
    assert decision.confidence == pytest.approx(0.97)
