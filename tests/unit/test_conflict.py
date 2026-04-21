"""Conflict-detection tests — now exercised through the canonical form layer.

Only pairs of values that both canonicalise to different canonical forms
trigger a conflict. Malformed VATs / TED notices in the `vat` field
canonicalise to None and therefore do NOT conflict.
"""

import pytest

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.company.exact_identifiers import ExactLeiMatch, ExactVatMatch
from src.consolidator.rules.conflict import find_conflict

LEI_A = "529900WTOG7RHO5TCH58"
LEI_B = "529900PC9XG1KHIJD788"
FR_VAT_A = "FR12345678901"
FR_VAT_B = "FR98765432109"


def _candidate(props: dict) -> Candidate:
    return Candidate(entity=Entity("Company", props["gmr_id"], props), context={})


def test_find_conflict_none_when_only_one_has_value():
    a = Entity("Company", "A", {"lei": LEI_A})
    b = _candidate({"gmr_id": "B", "vat": FR_VAT_A})
    assert find_conflict(a, b) is None


def test_find_conflict_none_when_values_match():
    a = Entity("Company", "A", {"lei": LEI_A, "vat": FR_VAT_A})
    b = _candidate({"gmr_id": "B", "lei": LEI_A, "vat": FR_VAT_A})
    assert find_conflict(a, b) is None


def test_find_conflict_hits_on_differing_canonical_lei():
    a = Entity("Company", "A", {"lei": LEI_A})
    b = _candidate({"gmr_id": "B", "lei": LEI_B})
    c = find_conflict(a, b)
    assert c is not None
    assert c[0] == "lei"


def test_find_conflict_prefers_lei_over_vat():
    a = Entity("Company", "A", {"lei": LEI_A, "vat": FR_VAT_A})
    b = _candidate({"gmr_id": "B", "lei": LEI_B, "vat": FR_VAT_B})
    c = find_conflict(a, b)
    assert c is not None
    assert c[0] == "lei"


def test_find_conflict_skips_non_canonical_vat():
    """Malformed 'VAT' (e.g. TED notice ID) canonicalises to None → no conflict,
    even if the other side is a valid VAT."""
    a = Entity("Company", "A", {"vat": "1594225-1-0-1"})       # TED notice
    b = _candidate({"gmr_id": "B", "vat": FR_VAT_A})            # valid FR VAT
    assert find_conflict(a, b) is None


def test_find_conflict_skips_malformed_on_both_sides():
    """A 'VAT' that's really a SIRET on one side and a TED notice on the
    other is not a conflict — neither canonicalises as a VAT."""
    a = Entity("Company", "A", {"vat": "83415751300815"})      # bare SIRET
    b = _candidate({"gmr_id": "B", "vat": "1594225-1-0-1"})    # TED notice
    assert find_conflict(a, b) is None


def test_find_conflict_handles_whitespace_same_vat():
    """`DE 273691032` and `DE273691032` canonicalise to the same value — no conflict."""
    a = Entity("Company", "A", {"vat": "DE 273691032"})
    b = _candidate({"gmr_id": "B", "vat": "DE273691032"})
    assert find_conflict(a, b) is None


def test_find_conflict_lei_whitespace_ignored():
    a = Entity("Company", "A", {"lei": f"  {LEI_A}  "})
    b = _candidate({"gmr_id": "B", "lei": LEI_A})
    assert find_conflict(a, b) is None


def test_authority_conflict_uses_authority_id():
    a = Entity("Authority", "A", {"authority_id": "auth-1"})
    b = Candidate(entity=Entity("Authority", "B", {"authority_id": "auth-2"}), context={})
    c = find_conflict(a, b)
    assert c is not None
    assert c[0] == "authority_id"


@pytest.mark.asyncio
async def test_exact_lei_refuses_when_vat_conflicts():
    """Same LEI but both VATs are valid AND different → conflict flag, no merge."""
    rule = ExactLeiMatch()
    entity = Entity("Company", "A", {"lei": LEI_A, "vat": FR_VAT_A})
    candidate = _candidate({"gmr_id": "B", "lei": LEI_A, "vat": FR_VAT_B})
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "flag"
    assert decision.details["conflict"] is True
    assert decision.details["conflicting_property"] == "vat"


@pytest.mark.asyncio
async def test_exact_lei_merges_when_one_vat_is_malformed():
    """Same LEI, one side has valid VAT, other has a TED-notice in `vat` →
    no real disagreement, merge should proceed."""
    rule = ExactLeiMatch()
    entity = Entity("Company", "A", {"lei": LEI_A, "vat": FR_VAT_A})
    candidate = _candidate({"gmr_id": "B", "lei": LEI_A, "vat": "1594225-1-0-1"})
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "merge"


@pytest.mark.asyncio
async def test_exact_vat_merges_when_no_conflict():
    rule = ExactVatMatch()
    entity = Entity("Company", "A", {"vat": FR_VAT_A, "name": "Acme"})
    candidate = _candidate({"gmr_id": "B", "vat": FR_VAT_A, "name": "Acme SAS"})
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "merge"
