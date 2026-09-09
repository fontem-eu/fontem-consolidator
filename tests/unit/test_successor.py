"""Unit tests for SuccessorLeiMatch — Neo4j mocked."""
# protected-access: the loader's `_loaded` / registry `_REGISTRY`
# are reset between tests so the per-test rule set is deterministic.
# import-outside-toplevel: loader / registry are imported inside
# tests so the reset above happens before module-import side effects.
# pylint: disable=protected-access,import-outside-toplevel

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.company.successor import (
    UNINFORMATIVE_LEGAL_FORM,
    SuccessorLeiMatch,
    corroborating_matches,
)


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
async def test_applies_only_when_entity_is_active_with_lei():
    rule = SuccessorLeiMatch()
    active = {"lei": "X", "name": "n", "country": "FR", "active": True}
    inactive = {"lei": "X", "name": "n", "country": "FR", "active": False}
    no_lei = {"name": "n", "country": "FR", "active": True}
    no_name = {"lei": "X", "country": "FR", "active": True}
    assert await rule.applies(Entity("Company", "A", active)) is True
    # inactive self → don't initiate
    assert await rule.applies(Entity("Company", "A", inactive)) is False
    # missing lei
    assert await rule.applies(Entity("Company", "A", no_lei)) is False
    # missing name
    assert await rule.applies(Entity("Company", "A", no_name)) is False


@pytest.mark.asyncio
async def test_resolve_carries_retired_lei_in_details():
    """The retired LEI must reach the action executor either way — it is
    what gets appended to the survivor's historic_leis before the merge,
    and losing it loses the lineage."""
    rule = SuccessorLeiMatch()
    entity = Entity("Company", "A", {
        "lei": "529900ACTIVEXXXXXXX1", "active": True,
        "postal_code": "811 09", "legal_form": "2EEG",
    })
    candidate = Candidate(
        entity=Entity("Company", "B", {
            "lei": "529900RETIREDXXXXXX2", "active": False,
            "postal_code": "811 09", "legal_form": "2EEG",
        }),
        context={"retired_lei": "529900RETIREDXXXXXX2"},
    )
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "merge"
    assert decision.rule_name == "successor_lei_match"
    assert decision.confidence == 0.98
    assert decision.details["retired_lei"] == "529900RETIREDXXXXXX2"
    assert decision.details["corroborated_on"] == ["postal_code"]


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


# ── corroboration: the change that replaced the LOU-prefix condition ──
#
# The old rule matched on name + country + active/inactive + the first
# four LEI characters (the issuing LOU), justified by a docstring claim
# that it used GLEIF's explicit successor links. It did not — we do not
# ingest them (src/etl/load_gleif.py extracts no successor field).
#
# Measured on shared 2026-09-07 over a 40,000-company sample: 124 pairs
# satisfy name + country + active/inactive. The LOU condition accepted
# 41 — rejecting 67% of genuine candidates, because an entity often
# re-registers precisely BECAUSE it moved LOU. Requiring a corroborating
# attribute instead accepts 83 and sends the other 41 to review.
#
# Every fixture below is a real pair from the shared graph.

def _co(gmr_id, **props):
    return Entity("Company", gmr_id, props)


def _pair(a_props, b_props):
    return _co("A", **a_props), Candidate(entity=_co("B", **b_props), context={})


# ---- corroborating_matches: the predicate in isolation ---------------

def test_postal_code_agreement_corroborates():
    """SWAN, a. s. / SWAN, a.s. (SVK) — same postal 811 09."""
    a, c = _pair(
        {"postal_code": "811 09", "legal_form": "2EEG"},
        {"postal_code": "811 09", "legal_form": "2EEG"},
    )
    assert corroborating_matches(a, c.entity) == ["postal_code"]


def test_postal_code_whitespace_is_normalised():
    """Trnavské Mýto, a.s. (SVK) — the same code written "82109" on one
    record and "821 09" on the other. Raw equality loses it."""
    a, c = _pair({"postal_code": "82109"}, {"postal_code": "821 09"})
    assert corroborating_matches(a, c.entity) == ["postal_code"]


def test_postal_code_case_is_normalised():
    """GBR codes appear in both cases across GLEIF records."""
    a, c = _pair({"postal_code": "dn14 6al"}, {"postal_code": "DN14 6AL"})
    assert corroborating_matches(a, c.entity) == ["postal_code"]


def test_legal_form_never_corroborates_because_it_classifies():
    """The regression that reached shared.

    legal_form was accepted as a corroborator. It is a CATEGORY: OV32 is
    the ELF code for an Italian S.R.L. and 157,598 Italian companies
    carry it, so "both are an S.R.L." corroborated every pair of
    same-named Italian companies in the country. 46 distinct
    "FUTURA S.R.L." records in Reggio Emilia, Ancona and Pisa were
    auto-merged into one entity at confidence 0.98, and 41
    "ALBA S.R.L." alongside them — 7,873 of 12,148 successor edges
    rested on legal_form alone.

    Excluding only the 8888 "unknown" code was the wrong cut: a real ELF
    code is exactly as non-discriminating as the unknown one. The test
    for a corroborator is not "is it populated" but "could two different
    companies share it".
    """
    a, c = _pair(
        {"postal_code": "42015", "legal_form": "OV32"},
        {"postal_code": "56029", "legal_form": "OV32"},
    )
    assert not corroborating_matches(a, c.entity)


def test_a_corroborator_must_distinguish_not_classify():
    """Pins the principle against the whole configured set, so adding a
    category attribute later fails here rather than in production."""
    from src.consolidator.rules.company.successor import (
        CORROBORATING_PROPERTIES)
    assert "legal_form" not in CORROBORATING_PROPERTIES
    assert set(CORROBORATING_PROPERTIES) == {
        "postal_code", "vat", "registered_as", "cik",
    }


def test_uninformative_legal_form_does_not_corroborate():  # noqa: D401
    """TKM GROUP PENSION SCHEME (GBR) — postcodes CR0 2LX vs CR0 2BX,
    both legal_form 8888. 8888 is GLEIF's "form not on the ELF list"
    and the single most common value in the graph (328,131 Companies),
    so two records sharing it is not evidence of anything."""
    a, c = _pair(
        {"postal_code": "CR0 2LX", "legal_form": UNINFORMATIVE_LEGAL_FORM},
        {"postal_code": "CR0 2BX", "legal_form": UNINFORMATIVE_LEGAL_FORM},
    )
    assert not corroborating_matches(a, c.entity)


def test_uninformative_legal_form_still_allows_postal_corroboration():
    """EDITH BIRKETT WILL TRUST (GBR) — 8888 on both, but the same
    postcode DN14 6AL. The postal agreement is real."""
    a, c = _pair(
        {"postal_code": "DN14 6AL", "legal_form": UNINFORMATIVE_LEGAL_FORM},
        {"postal_code": "DN14 6AL", "legal_form": UNINFORMATIVE_LEGAL_FORM},
    )
    assert corroborating_matches(a, c.entity) == ["postal_code"]


def test_missing_values_never_corroborate():
    """Absent is not agreement. Two nulls are not evidence."""
    assert not corroborating_matches(
        _co("A", postal_code=None, legal_form=None),
        _co("B", postal_code=None, legal_form=None),
    )
    a, c = _pair({"postal_code": "811 09"}, {})
    assert not corroborating_matches(a, c.entity)
    a, c = _pair({}, {"legal_form": "2EEG"})
    assert not corroborating_matches(a, c.entity)


def test_empty_string_never_corroborates():
    a, c = _pair(
        {"postal_code": "", "legal_form": ""},
        {"postal_code": "", "legal_form": ""},
    )
    assert not corroborating_matches(a, c.entity)


def test_whitespace_only_postal_never_corroborates():
    a, c = _pair({"postal_code": "   "}, {"postal_code": "   "})
    assert not corroborating_matches(a, c.entity)


def test_disagreeing_postal_codes_do_not_corroborate():
    """ENERGY OPTIMAL s.r.o. (SVK) — postal 82102 vs 83106 genuinely
    differ. Sharing the legal form VSZS does not rescue it; this pair
    goes to review."""
    a, c = _pair(
        {"postal_code": "82102", "legal_form": "VSZS"},
        {"postal_code": "83106", "legal_form": "VSZS"},
    )
    assert not corroborating_matches(a, c.entity)


def test_a_hard_identifier_corroborates():
    """vat / registered_as / cik do distinguish entities, so agreement
    on one is real evidence even when the addresses differ."""
    a, c = _pair(
        {"postal_code": "82102", "vat": "SK2020317068"},
        {"postal_code": "83106", "vat": "SK2020317068"},
    )
    assert corroborating_matches(a, c.entity) == ["vat"]


# ---- resolve(): which branch a pair lands in ------------------------

def _entity(**props):
    base = {"lei": "529900ACTIVEXXXXXXX1", "active": True, "country": "SVK",
            "name": "SWAN, a. s."}
    base.update(props)
    return Entity("Company", "A", base)


def _candidate(**props):
    base = {"lei": "097900BIFX0000156555", "active": False, "country": "SVK",
            "name": "SWAN, a.s."}
    base.update(props)
    return Candidate(entity=Entity("Company", "B", base),
                     context={"retired_lei": base["lei"]})


@pytest.mark.asyncio
async def test_corroborated_pair_merges():
    rule = SuccessorLeiMatch()
    d = await rule.resolve(
        _entity(postal_code="811 09", legal_form="2EEG"),
        _candidate(postal_code="811 09", legal_form="2EEG"),
    )
    assert d.action == "merge"
    assert d.confidence == 0.98
    assert "uncorroborated" not in d.details


@pytest.mark.asyncio
async def test_uncorroborated_pair_is_flagged_not_merged():
    """Same name, same country, one retired — but nothing else agrees.
    Most such pairs are still successions; some are a dissolved company
    and an unrelated one that took the name. A human decides."""
    rule = SuccessorLeiMatch()
    d = await rule.resolve(
        _entity(postal_code="811 09", legal_form="2EEG"),
        _candidate(postal_code="999 99", legal_form="VSZS"),
    )
    assert d.action == "flag"
    assert d.details["uncorroborated"] is True
    assert not d.details["corroborated_on"]


@pytest.mark.asyncio
async def test_uncorroborated_confidence_is_below_every_auto_merge_threshold():
    """force_auto_merge only fires on action == "merge", and
    auto_merge_threshold is None here — but the emitted confidence must
    also sort these to the top of a review queue ordered ascending, and
    stay under any threshold a future change might introduce."""
    rule = SuccessorLeiMatch()
    d = await rule.resolve(
        _entity(postal_code="A"), _candidate(postal_code="B"),
    )
    assert d.confidence == rule.UNCORROBORATED_CONFIDENCE
    assert d.confidence < 0.97
    assert d.confidence < rule.confidence


@pytest.mark.asyncio
async def test_uninformative_form_alone_routes_to_review():
    """The GBR trust shape: both 8888, postcodes differ. Under the old
    rule the LOU prefix matched and this merged silently."""
    rule = SuccessorLeiMatch()
    d = await rule.resolve(
        _entity(postal_code="CR0 2LX", legal_form=UNINFORMATIVE_LEGAL_FORM),
        _candidate(postal_code="CR0 2BX", legal_form=UNINFORMATIVE_LEGAL_FORM),
    )
    assert d.action == "flag"


@pytest.mark.asyncio
async def test_force_auto_merge_is_still_declared():
    """The corroborated branch relies on it: the engine honours
    force_auto_merge only when decision.action == "merge"."""
    assert SuccessorLeiMatch.force_auto_merge is True
    assert SuccessorLeiMatch.auto_merge_threshold is None, (
        "a threshold here would let the flag branch auto-merge on "
        "confidence, defeating the review routing"
    )


# ---- the query: what the LOU condition used to exclude ---------------

@pytest.mark.asyncio
async def test_query_no_longer_filters_on_the_lou_prefix():
    """The regression this change exists for.

    An entity frequently re-registers BECAUSE it moved LOU, so the
    predecessor and successor LEIs differ in exactly the first four
    characters the old rule required to be equal. Measured on shared:
    the condition rejected 83 of 124 genuine candidates (67%).
    """
    rule = SuccessorLeiMatch()
    driver = _fake_driver([])
    with patch("src.consolidator.neo4j.client.get_driver",
               AsyncMock(return_value=driver)):
        await rule.find_candidates(_entity())
    session = driver.session.return_value.__aenter__.return_value
    cypher = session.run.call_args.args[0]
    assert "left(" not in cypher, "still filtering on the LEI's LOU prefix"
    assert "b.lei <> $self_lei" in cypher, "must still exclude the same LEI"
    assert "coalesce(b.active, true) = false" in cypher
    assert "b.country = $country" in cypher
    assert "apoc.text.clean($name)" in cypher


@pytest.mark.asyncio
async def test_candidate_with_a_different_lou_is_now_returned():
    """SVK: 097900... (predecessor) vs 549300... (successor) — a real
    shared pair whose LEIs come from different LOUs."""
    rule = SuccessorLeiMatch()
    rec = {"b": {"gmr_id": "B", "lei": "097900BIFX0000156555",
                 "name": "ENERGY OPTIMAL s.r.o.", "active": False,
                 "postal_code": "82102", "legal_form": "VSZS"}}
    driver = _fake_driver([rec])
    with patch("src.consolidator.neo4j.client.get_driver",
               AsyncMock(return_value=driver)):
        out = await rule.find_candidates(
            _entity(lei="549300SUYKFINEFOZ839", name="ENERGY OPTIMAL s.r.o."))
    assert len(out) == 1
    assert out[0].context["retired_lei"] == "097900BIFX0000156555"
