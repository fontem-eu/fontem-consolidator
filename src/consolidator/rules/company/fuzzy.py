"""Fuzzy name rule — flag (:SAME_AS) candidates within the same country above a threshold.

Mirrors the existing match_unlinked.py logic: uses the Company.name full-text index
with Lucene char stripping, normalizes score by length, threshold 0.85.
"""

import re

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule

_LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _sanitize(name: str) -> str:
    return _LUCENE_SPECIAL.sub(" ", name).strip()


class FuzzyNameSameCountry(Rule):
    name = "fuzzy_name_same_country"
    description = (
        "Full-text name search against other :Company in same country; "
        f"normalised score ≥ {settings.fuzzy_name_threshold} → flag :SAME_AS for review."
    )
    entity_types = {"Company"}
    confidence = 0.9  # upper bound; emitted confidence varies per candidate
    action = "flag"

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name")) and bool(entity.properties.get("country"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        sanitized = _sanitize(entity.properties["name"])
        if not sanitized:
            return []
        async with driver.session() as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes('company_name_ft', $query)
                    YIELD node, score
                    WHERE node.gmr_id <> $self_id
                      AND node.country = $country
                    RETURN node, score
                    LIMIT 20
                    """,
                    query=sanitized,
                    self_id=entity.id,
                    country=entity.properties["country"],
                )
                records = [record async for record in result]
            except Exception:  # index may not exist in ephemeral test DBs
                return []
        threshold = settings.fuzzy_name_threshold
        name_len = max(1, len(entity.properties["name"]))
        out = []
        for rec in records:
            score = rec["score"]
            confidence = score / (name_len * 0.1 + 1)
            if confidence < threshold:
                continue
            props = dict(rec["node"])
            out.append(
                Candidate(
                    entity=Entity("Company", props["gmr_id"], props),
                    context={"raw_score": score, "normalised": confidence},
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
            entity_type="Company",
            details={"raw_score": candidate.context.get("raw_score")},
        )
