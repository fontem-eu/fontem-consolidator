"""Hard-identifier Company rules — LEI, CIK, VAT. All auto-merge when the same id appears on two nodes."""

from neo4j import AsyncDriver

from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class _ExactIdRule(Rule):
    """Shared logic: MATCH other :Company with same value for `id_property`."""

    entity_types = {"Company"}
    confidence = 1.0
    action = "merge"

    id_property: str  # subclass fills in
    _driver_factory = None  # injected at service startup

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get(self.id_property))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver: AsyncDriver = await get_driver()
        value = entity.properties[self.id_property]
        async with driver.session() as session:
            result = await session.run(
                f"""
                MATCH (c:Company)
                WHERE c.{self.id_property} = $value AND c.gmr_id <> $self_id
                RETURN c
                """,
                value=value,
                self_id=entity.id,
            )
            records = [record async for record in result]
        candidates = []
        for rec in records:
            props = dict(rec["c"])
            target = Entity(entity_type="Company", id=props["gmr_id"], properties=props)
            candidates.append(Candidate(entity=target, context={"matched_value": value}))
        return candidates

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        return Decision(
            rule_name=self.name,
            action="merge",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Company",
            details={"matched_property": self.id_property, **candidate.context},
        )


class ExactLeiMatch(_ExactIdRule):
    name = "exact_lei_match"
    description = "Two :Company nodes share the same LEI → merge."
    id_property = "lei"


class ExactCikMatch(_ExactIdRule):
    name = "exact_cik_match"
    description = "Two :Company nodes share the same SEC CIK → merge."
    id_property = "cik"


class ExactVatMatch(_ExactIdRule):
    name = "exact_vat_match"
    description = "Two :Company nodes share the same VAT → merge."
    id_property = "vat"
