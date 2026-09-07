"""Successor-LEI rule.

Recognises the "same real-world entity re-registered with a new LEI"
pattern: one node active, the other inactive, same normalised name and
country, plus at least one independent attribute agreeing.

What this rule does NOT have
---------------------------
GLEIF publishes an explicit successor link (``SuccessorLEI`` on an
expired record), but we do not ingest it — ``src/etl/load_gleif.py``
extracts name, address, legal form and status, and no successor field
reaches the graph. An earlier version of this docstring claimed the
rule used those links and justified ``force_auto_merge`` on that basis.
It did not: it matched on name + country + active/inactive + the first
four characters of the LEI. Ingesting the real successor link is the
proper fix and would make this rule deterministic; until then this is a
heuristic and is documented as one.

Why the LOU prefix is gone
--------------------------
The first four LEI characters are the issuing LOU. That condition was
doing almost no filtering — 5493 alone covers a large share of all LEIs
— while rejecting the very cases the rule exists for: an entity often
re-registers *because* it moved LOU, so the predecessor and successor
LEIs differ in exactly those four characters.

Measured on shared 2026-09-07 over a 40,000-company sample: 124 pairs
satisfy name + country + active/inactive, and the LOU condition
accepted only 41 of them. It rejected 67% of genuine candidates and
contributed no precision.

Corroboration instead
---------------------
A second attribute has to agree: postal code or legal form. That is a
real same-entity signal rather than a coincidence of issuer.

  * 83 of those 124 pairs (67%) are corroborated → auto-merge
  * the remaining 41 are flagged for review instead of merged
  * of the 41 the old LOU rule auto-merged, 5 had no corroboration at
    all — those stop being silent merges

So the change raises auto-merge recall 41 → 83 while removing the
uncorroborated merges. Postal code alone is not required: only 52% of
accepted pairs share one, because an entity that re-registers has
frequently also moved. Both corroborators are normalised first — see
UNINFORMATIVE_LEGAL_FORM and _normalise_postal, without which "8888"
(GLEIF's "form unknown") would corroborate 328,131 Companies with each
other and "821 09" would fail to match "82109".

The action executor, on seeing rule_name == 'successor_lei_match',
appends the retired LEI to the survivor's `historic_leis` array BEFORE
collapsing the nodes — so the lineage survives the merge.

Contrast with `exact_name_country_match`, which correctly refuses to
merge when two ACTIVE entities share a name but have different LEIs
(the CAISSE / sibling case). The active/inactive asymmetry is what
separates "predecessor" from "sibling", and it is load-bearing here.
"""

from src.consolidator.rules.base import Candidate, Decision, Entity, Rule

#: GLEIF's ELF code for "entity legal form not on the ELF code list" —
#: i.e. unknown. It is the single most common value in the graph
#: (328,131 Companies in shared, ahead of every real form), so two
#: entities sharing it is not evidence of anything and must not count
#: as corroboration. Every GBR trust carries it.
UNINFORMATIVE_LEGAL_FORM = "8888"


def _normalise_postal(value: str | None) -> str | None:
    """Postal codes for one address are written inconsistently across
    GLEIF records — "82109" and "821 09" are the same Slovak code, and
    comparing them raw loses real corroboration (8 pairs of 181 in
    shared). Case-fold and drop whitespace."""
    if not value:
        return None
    stripped = "".join(value.split()).upper()
    return stripped or None


def _informative_legal_form(value: str | None) -> str | None:
    if not value or value == UNINFORMATIVE_LEGAL_FORM:
        return None
    return value


#: Attributes that can corroborate a successor match, each with the
#: normaliser that decides what "agrees" means for it. A property whose
#: normaliser returns None contributes nothing.
CORROBORATING_PROPERTIES: dict[str, object] = {
    "postal_code": _normalise_postal,
    "legal_form": _informative_legal_form,
}


def corroborating_matches(entity: Entity, candidate: Entity) -> list[str]:
    """Which corroborating attributes are present on both and agree.

    A missing value never corroborates: absent is not agreement, and two
    nulls are not evidence. Neither is a value the normaliser rejects as
    uninformative.
    """
    agreed = []
    for prop, normalise in CORROBORATING_PROPERTIES.items():
        mine = normalise(entity.properties.get(prop))
        theirs = normalise(candidate.properties.get(prop))
        if mine and theirs and mine == theirs:
            agreed.append(prop)
    return agreed


class SuccessorLeiMatch(Rule):
    name = "successor_lei_match"
    description = (
        "Active + inactive :Company with the same normalised name and "
        "country, corroborated by postal code or legal form → merge. "
        "Uncorroborated pairs are flagged for review. The retired LEI is "
        "preserved on the survivor's `historic_leis` array."
    )
    entity_types = {"Company"}
    confidence = 0.98
    action = "merge"
    # Applies only to the corroborated branch: the engine honours
    # force_auto_merge when decision.action == "merge", and the
    # uncorroborated branch returns "flag". auto_merge_threshold stays
    # None so a flag from this rule can never auto-merge on confidence.
    force_auto_merge = True

    #: Emitted on the uncorroborated branch. Below every auto-merge
    #: threshold in the system, and low enough to sort to the top of a
    #: review queue ordered by ascending confidence.
    UNCORROBORATED_CONFIDENCE = 0.80

    async def applies(self, entity: Entity) -> bool:
        # We only initiate successor consolidation from the ACTIVE side.
        # An inactive node getting consolidated will be handled when the
        # corresponding active node runs its pipeline.
        lei = entity.properties.get("lei")
        active = entity.properties.get("active")
        return bool(lei) and bool(entity.properties.get("name")) and bool(
            entity.properties.get("country")
        ) and active is True

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        # Imported lazily so unit tests patching the module-level
        # `get_driver` see the patched callable rather than a name
        # already bound at import time.
        from src.consolidator.neo4j.client import get_driver  # pylint: disable=import-outside-toplevel
        driver = await get_driver()
        async with driver.session() as session:
            # name_clean is materialised by the Neo4j sink at
            # projection time; the range index keeps this O(log N)
            # rather than full-scanning 3.5M Companies.
            result = await session.run(
                """
                MATCH (b:Company)
                WHERE b.name_clean = apoc.text.clean($name)
                  AND b.country = $country
                  AND b.gmr_id <> $self_id
                  AND b.lei IS NOT NULL
                  AND b.lei <> $self_lei
                  AND coalesce(b.active, true) = false
                RETURN b
                """,
                name=entity.properties["name"],
                country=entity.properties["country"],
                self_id=entity.id,
                self_lei=entity.properties["lei"],
            )
            records = [record async for record in result]
        return [
            Candidate(
                entity=Entity("Company", dict(rec["b"])["gmr_id"], dict(rec["b"])),
                context={"retired_lei": dict(rec["b"]).get("lei")},
            )
            for rec in records
        ]

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        agreed = corroborating_matches(entity, candidate.entity)
        details = {
            "retired_lei": candidate.context.get("retired_lei"),
            "corroborated_on": agreed,
        }
        if not agreed:
            # Same name, same country, one retired — but nothing else
            # agrees. Most are still successions; some are a dissolved
            # company and an unrelated one that took the name. A human
            # decides.
            return Decision(
                rule_name=self.name,
                action="flag",
                source_id=entity.id,
                target_id=candidate.entity.id,
                confidence=self.UNCORROBORATED_CONFIDENCE,
                entity_type="Company",
                details={**details, "uncorroborated": True},
            )
        return Decision(
            rule_name=self.name,
            action="merge",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Company",
            details=details,
        )
