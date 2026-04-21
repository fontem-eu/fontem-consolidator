"""Conflict-detection helper tests — and the integration that exact-id rules now use it."""

import pytest

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.company.exact_identifiers import ExactLeiMatch, ExactVatMatch
from src.consolidator.rules.conflict import find_conflict


def _candidate(props: dict) -> Candidate:
    return Candidate(entity=Entity("Company", props["gmr_id"], props), context={})


def test_find_conflict_none_when_only_one_has_value():
    a = Entity("Company", "A", {"lei": "LEI-A"})
    b = _candidate({"gmr_id": "B", "vat": "FR-123"})  # b has vat, a does not
    assert find_conflict(a, b) is None


def test_find_conflict_none_when_values_match():
    a = Entity("Company", "A", {"lei": "LEI-X", "vat": "FR-1"})
    b = _candidate({"gmr_id": "B", "lei": "LEI-X", "vat": "FR-1"})
    assert find_conflict(a, b) is None


def test_find_conflict_hits_on_differing_lei():
    a = Entity("Company", "A", {"lei": "LEI-A"})
    b = _candidate({"gmr_id": "B", "lei": "LEI-B"})
    c = find_conflict(a, b)
    assert c is not None
    assert c[0] == "lei"


def test_find_conflict_prefers_lei_over_vat():
    a = Entity("Company", "A", {"lei": "L1", "vat": "V1"})
    b = _candidate({"gmr_id": "B", "lei": "L2", "vat": "V2"})
    c = find_conflict(a, b)
    assert c is not None
    assert c[0] == "lei"  # lei is first in the tuple order


def test_authority_conflict_uses_authority_id():
    a = Entity("Authority", "A", {"authority_id": "auth-1"})
    b = Candidate(entity=Entity("Authority", "B", {"authority_id": "auth-2"}), context={})
    c = find_conflict(a, b)
    assert c is not None
    assert c[0] == "authority_id"


@pytest.mark.asyncio
async def test_exact_lei_refuses_when_vat_conflicts():
    """Two companies share an LEI but have different VATs → conflict flag, no merge."""
    rule = ExactLeiMatch()
    entity = Entity("Company", "A", {"lei": "LEI-X", "vat": "FR-1"})
    candidate = _candidate({"gmr_id": "B", "lei": "LEI-X", "vat": "FR-2"})
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "flag"
    assert decision.details["conflict"] is True
    assert decision.details["conflicting_property"] == "vat"


@pytest.mark.asyncio
async def test_exact_vat_merges_when_no_conflict():
    rule = ExactVatMatch()
    entity = Entity("Company", "A", {"vat": "FR-123", "name": "Acme"})
    candidate = _candidate({"gmr_id": "B", "vat": "FR-123", "name": "Acme SAS"})
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "merge"
