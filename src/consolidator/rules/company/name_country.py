from src.consolidator.rules.base import Candidate, Decision, Entity, Rule
from src.consolidator.rules.conflict import conflict_decision, find_conflict


class ExactNameCountryMatch(Rule):
    name = "exact_name_country_match"
    description = (
        "Two :Company nodes share the same (name, country) after whitespace/"
        "punctuation normalisation (apoc.text.clean) → merge iff no hard-id conflict."
    )
    entity_types = {"Company"}
    confidence = 0.95
    action = "merge"

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name")) and bool(entity.properties.get("country"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        async with driver.session() as session:
            # apoc.text.clean strips non-alphanumerics + lowercases, so
            # "NEURAXPHARM FRANCE ( Rang 1)" == "NEURAXPHARM FRANCE (Rang 1)"
            # == "NEURAXPHARM FRANCE  (Rang 1)" all collapse.
            result = await session.run(
                """
                MATCH (c:Company)
                WHERE apoc.text.clean(c.name) = apoc.text.clean($name)
                  AND c.country = $country
                  AND c.gmr_id <> $self_id
                RETURN c
                """,
                name=entity.properties["name"],
                country=entity.properties["country"],
                self_id=entity.id,
            )
            records = [record async for record in result]
        out = []
        for rec in records:
            props = dict(rec["c"])
            out.append(Candidate(entity=Entity("Company", props["gmr_id"], props), context={}))
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        # Refuse the auto-merge and downgrade to a conflict-flag when any hard id disagrees.
        conflict = find_conflict(entity, candidate)
        if conflict:
            return conflict_decision(
                rule_name=self.name,
                entity=entity,
                candidate=candidate,
                confidence=self.confidence,
                conflict=conflict,
            )
        return Decision(
            rule_name=self.name,
            action="merge",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Company",
            details={},
        )
