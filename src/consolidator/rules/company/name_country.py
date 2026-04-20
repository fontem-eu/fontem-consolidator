from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class ExactNameCountryMatch(Rule):
    name = "exact_name_country_match"
    description = "Two :Company nodes share the same normalized (name, country) → merge iff no LEI/VAT conflict."
    entity_types = {"Company"}
    confidence = 0.95
    action = "merge"

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name")) and bool(entity.properties.get("country"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Company)
                WHERE toLower(c.name) = toLower($name)
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
        # Conflict detection: if both have LEI/VAT and they differ, refuse the auto-merge
        # and downgrade to a conflict-flag via the engine + actions layer.
        for prop in ("lei", "vat", "cik"):
            a = entity.properties.get(prop)
            b = candidate.entity.properties.get(prop)
            if a and b and a != b:
                return Decision(
                    rule_name=self.name,
                    action="flag",
                    source_id=entity.id,
                    target_id=candidate.entity.id,
                    confidence=self.confidence,
                    entity_type="Company",
                    details={
                        "conflict": True,
                        "conflicting_property": prop,
                        "left": a,
                        "right": b,
                    },
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
