"""A name match resting on one word must not auto-merge.

_normalise strips the legal form, so "Siemens AG" and a bare "Siemens"
both become "SIEMENS" and score Jaro-Winkler 1.0 — above the 0.97
auto_merge_threshold. That is right for "ALCON NV" / "ALCON" and wrong
for a record that just says "Siemens": Siemens Energy AG and Siemens
Healthineers AG have been separately listed companies since 2020, and a
bare brand token cannot say which one it meant. Attributing another
listed company's contracts to Siemens AG is not a recoverable error —
nothing later contradicts it.

Measured on shared 2026-09-07: 25 of 671 auto-approved pairs (3.7%)
had a single-token name with no hard identifier on at least one side.
All 25 looked correct on inspection — mostly case variants
("TRADIUM"/"Tradium") and legal-form strips ("ALCON NV"/"ALCON") — so
this is about the shape being unverifiable, not about a known bad
merge. They become review items that sort to the front of the queue
rather than silent merges.
"""
# pylint: disable=protected-access
import pytest

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.company.fuzzy import (
    SINGLE_TOKEN_CONFIDENCE,
    FuzzyNameSameCountry,
    _normalise,
    _rests_on_one_token,
)
from src.config import settings


def _pair(a_name, b_name, a_props=None, b_props=None, sim=1.0):
    a = Entity("Company", "A", {"name": a_name, "country": "DEU", **(a_props or {})})
    b = Entity("Company", "B", {"name": b_name, "country": "DEU", **(b_props or {})})
    return a, Candidate(entity=b, context={"jw_similarity": sim})


# ── the predicate ────────────────────────────────────────────────────

def test_legal_form_stripping_collapses_a_qualified_name_to_a_bare_one():
    """The mechanism the discount exists for."""
    assert _normalise("Siemens AG") == "SIEMENS"
    assert _normalise("Siemens") == "SIEMENS"


def test_single_token_with_no_identifiers_rests_on_one_token():
    assert _rests_on_one_token("SIEMENS", {}, {}) is True


def test_identifiers_on_both_sides_corroborate():
    """find_conflict already rejects the pair if hard IDs disagree, so
    both sides carrying one is evidence the name alone cannot give."""
    assert _rests_on_one_token("SIEMENS", {"lei": "X"}, {"vat": "Y"}) is False


def test_identifier_on_only_one_side_is_not_corroboration():
    """"Siemens AG" with a LEI matched to a bare "Siemens" with nothing
    is exactly the risky shape — the LEI says nothing about the record
    that has none."""
    assert _rests_on_one_token("SIEMENS", {"lei": "X"}, {}) is True
    assert _rests_on_one_token("SIEMENS", {}, {"lei": "X"}) is True


def test_multi_token_names_are_never_discounted():
    """SANOFI AVENTIS SA (ESP) sat at the centre of a 7-record cluster of
    punctuation variants. Two distinctive words is not a bare brand, and
    those clusters must keep auto-merging."""
    assert _rests_on_one_token("SANOFI AVENTIS", {}, {}) is False
    assert _normalise("SANOFI AVENTIS SA") == "SANOFI AVENTIS"


@pytest.mark.parametrize("identifier", ["lei", "vat", "cik", "registered_as"])
def test_any_hard_identifier_counts(identifier):
    assert _rests_on_one_token("SIEMENS", {identifier: "V"}, {identifier: "W"}) is False


def test_empty_identifier_values_do_not_count():
    assert _rests_on_one_token("SIEMENS", {"lei": ""}, {"lei": None}) is True


# ── resolve(): the emitted confidence ────────────────────────────────

@pytest.mark.asyncio
async def test_bare_brand_match_is_capped_below_auto_merge():
    rule = FuzzyNameSameCountry()
    entity, candidate = _pair("Siemens AG", "Siemens", sim=1.0)
    d = await rule.resolve(entity, candidate)
    assert d.action == "flag"
    assert d.confidence == SINGLE_TOKEN_CONFIDENCE
    assert d.confidence < rule.auto_merge_threshold, (
        "a capped score at or above the threshold still auto-merges"
    )
    assert d.details["single_token_match"] is True
    assert d.details["uncorroborated_token"] == "SIEMENS"
    # the real similarity is preserved for the reviewer
    assert d.details["jw_similarity"] == 1.0


@pytest.mark.asyncio
async def test_capped_score_stays_inside_the_review_band():
    """Below fuzzy_name_threshold the pair is dropped entirely and never
    reaches a reviewer, which would be worse than auto-merging it."""
    assert SINGLE_TOKEN_CONFIDENCE >= settings.fuzzy_name_threshold
    assert SINGLE_TOKEN_CONFIDENCE < FuzzyNameSameCountry.auto_merge_threshold


@pytest.mark.asyncio
async def test_capped_score_sorts_ahead_of_ordinary_review_items():
    """"Pops out first" is the point: an ascending-confidence queue must
    put these before a routine 0.95 fuzzy flag."""
    rule = FuzzyNameSameCountry()
    bare = await rule.resolve(*_pair("Siemens AG", "Siemens", sim=1.0))
    ordinary = await rule.resolve(
        *_pair("Mercedes-Benz Leasing GmbH", "Mercedes-Benz Leasing", sim=0.95))
    assert bare.confidence < ordinary.confidence


@pytest.mark.asyncio
async def test_multi_token_match_keeps_its_similarity_and_auto_merges():
    rule = FuzzyNameSameCountry()
    d = await rule.resolve(
        *_pair("SANOFI AVENTIS SA", "SANOFI-AVENTIS, S.A.", sim=0.99))
    assert d.confidence == 0.99
    assert d.confidence >= rule.auto_merge_threshold
    assert "single_token_match" not in d.details


@pytest.mark.asyncio
async def test_corroborated_single_token_still_auto_merges():
    """"ALCON NV" / "ALCON" with identifiers on both sides is the benign
    shape and should not be penalised."""
    rule = FuzzyNameSameCountry()
    d = await rule.resolve(*_pair(
        "ALCON NV", "ALCON",
        a_props={"lei": "5493001"}, b_props={"vat": "BE0123"},
        sim=1.0,
    ))
    assert d.confidence == 1.0
    assert "single_token_match" not in d.details


@pytest.mark.asyncio
async def test_a_low_similarity_single_token_pair_is_not_raised():
    """The cap is a ceiling, never a floor — min(), not assignment."""
    rule = FuzzyNameSameCountry()
    d = await rule.resolve(*_pair("Siemens AG", "Siemens", sim=0.925))
    assert d.confidence == 0.925
