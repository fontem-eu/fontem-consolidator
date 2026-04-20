"""Authority-entity rules. Authorities are TED buyers (public bodies)."""

import re

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule

_LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


class ExactAuthorityIdMatch(Rule):
    name = "exact_authority_id_match"
    description = "Two :Authority nodes share the same authority_id → merge (usually a no-op; unique constraint guards it)."
    entity_types = {"Authority"}
    confidence = 1.0
    action = "merge"

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("authority_id"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (a:Authority)
                WHERE a.authority_id = $value AND a.authority_id <> $self_id
                RETURN a
                """,
                value=entity.properties["authority_id"],
                self_id=entity.id,
            )
            records = [record async for record in result]
        out = []
        for rec in records:
            props = dict(rec["a"])
            out.append(
                Candidate(entity=Entity("Authority", props["authority_id"], props), context={})
            )
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        return Decision(
            rule_name=self.name,
            action="merge",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Authority",
            details={},
        )


class ExactNameCountryMatchAuthority(Rule):
    name = "exact_name_country_match_authority"
    description = "Two :Authority with same (name, country) → merge."
    entity_types = {"Authority"}
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
                MATCH (a:Authority)
                WHERE toLower(a.name) = toLower($name)
                  AND a.country = $country
                  AND a.authority_id <> $self_id
                RETURN a
                """,
                name=entity.properties["name"],
                country=entity.properties["country"],
                self_id=entity.id,
            )
            records = [record async for record in result]
        out = []
        for rec in records:
            props = dict(rec["a"])
            out.append(
                Candidate(entity=Entity("Authority", props["authority_id"], props), context={})
            )
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        return Decision(
            rule_name=self.name,
            action="merge",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Authority",
            details={},
        )


class FuzzyNameSameCountryAuthority(Rule):
    name = "fuzzy_name_same_country_authority"
    description = (
        f"Full-text name match across :Authority in same country, "
        f"threshold {settings.fuzzy_name_threshold} → flag :SAME_AS."
    )
    entity_types = {"Authority"}
    confidence = 0.9
    action = "flag"

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name")) and bool(entity.properties.get("country"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        sanitized = _LUCENE_SPECIAL.sub(" ", entity.properties["name"]).strip()
        if not sanitized:
            return []
        async with driver.session() as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes('authority_name_ft', $query)
                    YIELD node, score
                    WHERE node.authority_id <> $self_id AND node.country = $country
                    RETURN node, score LIMIT 20
                    """,
                    query=sanitized,
                    self_id=entity.id,
                    country=entity.properties["country"],
                )
                records = [record async for record in result]
            except Exception:
                return []
        out = []
        name_len = max(1, len(entity.properties["name"]))
        for rec in records:
            normalised = rec["score"] / (name_len * 0.1 + 1)
            if normalised < settings.fuzzy_name_threshold:
                continue
            props = dict(rec["node"])
            out.append(
                Candidate(
                    entity=Entity("Authority", props["authority_id"], props),
                    context={"raw_score": rec["score"], "normalised": normalised},
                )
            )
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=min(0.95, candidate.context.get("normalised", self.confidence)),
            entity_type="Authority",
            details={"raw_score": candidate.context.get("raw_score")},
        )
