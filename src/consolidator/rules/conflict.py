"""Shared conflict detection.

When a hard-identifier rule (LEI, CIK, VAT, authority_id) finds a candidate,
but the candidate has a *different* value on another hard identifier, that's
a data-integrity signal that the two nodes may represent different real
entities. Refuse the auto-merge and emit a conflict flag for human review.
"""

from src.consolidator.rules.base import Candidate, Decision, Entity

# Properties whose values must match if both nodes have them. Ordered by strength.
_HARD_IDS_BY_ENTITY: dict[str, tuple[str, ...]] = {
    "Company": ("lei", "cik", "vat"),
    "Authority": ("authority_id",),
}


def find_conflict(entity: Entity, candidate: Candidate) -> tuple[str, object, object] | None:
    """Return (property, left_value, right_value) for the first conflicting hard id,
    or None if all hard ids agree (or only one side has the id)."""
    for prop in _HARD_IDS_BY_ENTITY.get(entity.entity_type, ()):
        a = entity.properties.get(prop)
        b = candidate.entity.properties.get(prop)
        if a and b and a != b:
            return prop, a, b
    return None


def conflict_decision(
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
