"""Successor-LEI rule.

Recognises the "same real-world entity re-registered with a new LEI"
pattern: one node is active, the other is inactive, they share a
normalised name + country + GLEIF LOU prefix (first 4 chars of the LEI).

Fires as a high-confidence auto-merge. The action executor, on seeing
rule_name == 'successor_lei_match', appends the retired LEI to the
survivor's `historic_leis` array BEFORE collapsing the nodes — so the
lineage survives the merge.

Contrast with `exact_name_country_match`, which correctly refuses to
merge when two ACTIVE entities share a name but have different LEIs
(the CAISSE / sibling case).
"""

from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class SuccessorLeiMatch(Rule):
    name = "successor_lei_match"
    description = (
        "Active + inactive :Company with same normalised name, country, and "
        "GLEIF LOU prefix → merge. The retired LEI is preserved on the "
        "survivor's `historic_leis` array."
    )
    entity_types = {"Company"}
    confidence = 0.98
    action = "merge"

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
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (b:Company)
                WHERE apoc.text.clean(b.name) = apoc.text.clean($name)
                  AND b.country = $country
                  AND b.gmr_id <> $self_id
                  AND b.lei IS NOT NULL
                  AND coalesce(b.active, true) = false
                  AND left(b.lei, 4) = left($self_lei, 4)
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
        return Decision(
            rule_name=self.name,
            action="merge",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Company",
            details={"retired_lei": candidate.context.get("retired_lei")},
        )
