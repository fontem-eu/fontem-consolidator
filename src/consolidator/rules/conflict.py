"""Shared conflict detection.

When a hard-identifier rule (LEI, CIK, VAT, authority_id) finds a candidate,
but the candidate has a *different* value on another hard identifier that
IS ALSO a valid instance of that identifier type, that's a data-integrity
signal that the two nodes may represent different real entities. Refuse
the auto-merge and emit a conflict flag for human review.

We canonicalise via `identifiers.canon_*` before comparing — so malformed
VATs, TED notice IDs in the `vat` field, and whitespace variations fall
out as `None` and don't trigger a false-positive conflict.
"""

from collections.abc import Callable

from src.consolidator import identifiers
from src.consolidator.rules.base import Candidate, Decision, Entity

# Property + canonicaliser pairs. Order matters — first conflict wins,
# stronger identifiers come first.
_HARD_IDS_BY_ENTITY: dict[str, tuple[tuple[str, Callable[[object], str | None]], ...]] = {
    "Company": (
        ("lei", identifiers.canon_lei),
        ("cik", identifiers.canon_cik),
        ("vat", identifiers.canon_vat),
    ),
    "Authority": (
        # authority_id isn't format-validated — compare raw with whitespace stripped
        ("authority_id", lambda v: str(v).strip() if v else None),
    ),
}

# WEAK identifiers: discriminating enough to block a merge, but only when
# no strong identifier vouched for the pair first.
#
# `registered_as` is the national registration number (Handelsregister,
# GSTIN, charity number, ...). It is by far the best-populated hard
# identifier we hold and it was NOT being checked, which is how 117,967
# same-name pairs with provably different registration numbers — different
# companies, frequently in different cities — passed conflict detection.
#
# It is weak rather than strong for one reason: two records for the SAME
# entity can legitimately carry different values here, because sources
# disagree about *which* registry identifier to record. So a strong
# identifier (LEI/CIK/VAT) that AGREES outranks it — see find_conflict.
_WEAK_IDS_BY_ENTITY: dict[str, tuple[tuple[str, Callable[[object], str | None]], ...]] = {
    "Company": (
        ("registered_as", identifiers.canon_registration_number),
    ),
}


def find_conflict(entity: Entity, candidate: Candidate) -> tuple[str, object, object] | None:
    """Return (property, left_canonical, right_canonical) when both sides have
    valid canonical forms of the same identifier and those forms disagree.
    Return None when they agree, one side is missing, or either side is
    non-canonical (malformed VAT, wrong-length LEI, etc.).

    Two tiers. A STRONG identifier that disagrees is decisive and returns
    immediately. A WEAK one (the national registration number) is only
    consulted when no strong identifier agreed: sharing an LEI makes two
    records the same legal entity by definition, and a registration number
    that differs there means the sources recorded different registry
    identifiers, not different companies.
    """
    strong_agreement = False
    for prop, canon in _HARD_IDS_BY_ENTITY.get(entity.entity_type, ()):
        a = canon(entity.properties.get(prop))
        b = canon(candidate.entity.properties.get(prop))
        if a and b:
            if a != b:
                return prop, a, b
            strong_agreement = True

    if strong_agreement:
        return None

    for prop, canon in _WEAK_IDS_BY_ENTITY.get(entity.entity_type, ()):
        a = canon(entity.properties.get(prop))
        b = canon(candidate.entity.properties.get(prop))
        if a and b and a != b:
            return prop, a, b
    return None


# Six kwargs mirror the conflict-flag :Decision shape (rule, source,
# candidate, score, conflict tuple, optional matched_property).
# Bundling them would just push the same columns into a struct.
def conflict_decision(  # pylint: disable=too-many-arguments
    *,
    rule_name: str,
    entity: Entity,
    candidate: Candidate,
    confidence: float,
    conflict: tuple[str, object, object],
    matched_property: str | None = None,
) -> Decision:
    """Build a flag+conflict Decision when find_conflict returned a hit."""
    prop, left, right = conflict
    details = {
        "conflict": True,
        "conflicting_property": prop,
        "left": left,
        "right": right,
    }
    if matched_property:
        details["matched_property"] = matched_property
    return Decision(
        rule_name=rule_name,
        action="flag",
        source_id=entity.id,
        target_id=candidate.entity.id,
        confidence=confidence,
        entity_type=entity.entity_type,
        details=details,
    )
