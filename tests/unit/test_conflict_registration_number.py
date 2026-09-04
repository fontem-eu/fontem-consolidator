"""Two-tier conflict detection: strong identifiers vs registration numbers.

`registered_as` (Handelsregister number, GSTIN, charity number, ...) is
the best-populated hard identifier in the graph and was not being
checked at all. That gap let 117,967 same-name pairs through conflict
detection whose registration numbers provably disagree — verified in
prod as distinct companies, frequently in different cities:

    Weigel GmbH   HRB 42909 (Offenbach)  vs  HRB 15993 (Nürnberg)
    BHARAT ELECTRICALS  GSTIN state 09 (UP) vs state 36 (Telangana)

It is a WEAK signal though: two records for the same entity can carry
different values here because sources disagree about which registry
identifier to record. So an agreeing strong identifier outranks it.
"""

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.conflict import find_conflict

LEI_A = "529900T8BM49AURSDO55"
LEI_B = "213800QILIUD4ROSUO03"


def _pair(props_a: dict, props_b: dict, entity_type: str = "Company"):
    entity = Entity(entity_type=entity_type, id="A", properties=props_a)
    candidate = Candidate(
        entity=Entity(entity_type=entity_type, id="B", properties=props_b),
        context={},
    )
    return entity, candidate


def test_differing_registration_numbers_conflict():
    entity, candidate = _pair(
        {"registered_as": "HRB 42909"}, {"registered_as": "HRB 15993"}
    )
    found = find_conflict(entity, candidate)
    assert found is not None
    prop, left, right = found
    assert prop == "registered_as"
    assert left == "HRB42909"
    assert right == "HRB15993"


def test_registration_numbers_compare_after_normalisation():
    """Formatting differences between sources are not disagreement."""
    entity, candidate = _pair(
        {"registered_as": "HRB 117457"}, {"registered_as": "hrb-117457"}
    )
    assert find_conflict(entity, candidate) is None


def test_agreeing_strong_id_outranks_registration_mismatch():
    """Sharing an LEI makes two records the same legal entity by
    definition; a differing registration number there means the sources
    recorded different registry identifiers, not different companies."""
    entity, candidate = _pair(
        {"lei": LEI_A, "registered_as": "HRB 42909"},
        {"lei": LEI_A, "registered_as": "12345678"},
    )
    assert find_conflict(entity, candidate) is None


def test_disagreeing_strong_id_still_wins():
    entity, candidate = _pair(
        {"lei": LEI_A, "registered_as": "HRB 42909"},
        {"lei": LEI_B, "registered_as": "HRB 42909"},
    )
    found = find_conflict(entity, candidate)
    assert found is not None
    assert found[0] == "lei"


def test_missing_registration_on_one_side_is_not_conflict():
    entity, candidate = _pair({"registered_as": "HRB 42909"}, {})
    assert find_conflict(entity, candidate) is None


def test_too_short_registration_is_ignored():
    """A 1-2 character value is a placeholder, not a registration."""
    entity, candidate = _pair({"registered_as": "1"}, {"registered_as": "2"})
    assert find_conflict(entity, candidate) is None


def test_authorities_have_no_weak_tier():
    """Only Company carries registered_as; the Authority path is
    unchanged by the two-tier split."""
    entity, candidate = _pair(
        {"authority_id": "X1", "registered_as": "HRB 1"},
        {"authority_id": "X1", "registered_as": "HRB 2"},
        entity_type="Authority",
    )
    assert find_conflict(entity, candidate) is None
