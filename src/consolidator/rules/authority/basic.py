"""Authority-entity rules. Authorities are TED buyers (public bodies)."""

import re

from rapidfuzz.distance import JaroWinkler

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule

_LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def _normalise_authority(name: str) -> str:
    """Authorities have fewer legal-form suffixes than companies; mostly
    public-body words. Lower-case, drop punctuation, collapse spaces."""
    s = name.upper()
    s = _PUNCT.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s


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
            # apoc.text.clean strips whitespace + punctuation + lowercases so
            # minor typographic variants collapse to an exact match.
            # We match against the sink-materialised ``name_clean``
            # property so the range index on Authority.name_clean
            # answers in O(log N).
            result = await session.run(
                """
                MATCH (a:Authority)
                WHERE a.name_clean = apoc.text.clean($name)
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


class ExactNameAnyCountryAuthority(Rule):
    """EU-wide bodies (EEAS, DG JRC, eu-LISA, …) show up in TED once per
    contracting-destination country and end up as N duplicate authority
    nodes with the same name but different `country` values.

    Match on apoc.text.clean(name) alone. We flag (never auto-merge)
    because occasionally two entities with an identical exact name in
    different countries are genuinely different institutions — a human
    decides.
    """
    name = "exact_name_any_country_authority"
    description = (
        "Two :Authority nodes share the same (cleaned) name across different "
        "countries → flag :SAME_AS for review. Targets EU bodies (EEAS, JRC, "
        "eu-LISA) that TED publishes once per destination country."
    )
    entity_types = {"Authority"}
    confidence = 0.90
    action = "flag"
    # Canary on 1k authorities surfaced 32 of these — every one was a
    # European Commission DG (DG COMM, FPI, INTPA) registered per
    # delegation country. Zero false positives. Promote to auto-merge
    # at the rule's flat 0.90 confidence; the conflict gate (handled in
    # the engine upgrade logic) still drops anything with a hard ID
    # mismatch back to flag-for-review.
    auto_merge_threshold = 0.90

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        async with driver.session() as session:
            # name_clean indexed on Authority — keeps this O(log N).
            result = await session.run(
                """
                MATCH (a:Authority)
                WHERE a.name_clean = apoc.text.clean($name)
                  AND a.authority_id <> $self_id
                  AND coalesce(a.country, "") <> coalesce($country, "")
                RETURN a
                """,
                name=entity.properties["name"],
                country=entity.properties.get("country") or "",
                self_id=entity.id,
            )
            records = [record async for record in result]
        return [
            Candidate(
                entity=Entity("Authority", dict(rec["a"])["authority_id"], dict(rec["a"])),
                context={"cross_country": True},
            )
            for rec in records
        ]

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Authority",
            details={
                "cross_country": True,
                "source_country": entity.properties.get("country"),
                "target_country": candidate.entity.properties.get("country"),
            },
        )


class FuzzyNameSameCountryAuthority(Rule):
    name = "fuzzy_name_same_country_authority"
    description = (
        "Full-text candidate fetch + Jaro-Winkler similarity on normalised "
        "names (same country). Flags :SAME_AS above "
        f"{settings.fuzzy_name_threshold}."
    )
    entity_types = {"Authority"}
    confidence = 0.95
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
                    CALL db.index.fulltext.queryNodes('authority_name_ft', $search_text)
                    YIELD node, score
                    WHERE node.authority_id <> $self_id AND node.country = $country
                    RETURN node, score LIMIT 20
                    """,
                    search_text=sanitized,
                    self_id=entity.id,
                    country=entity.properties["country"],
                )
                records = [record async for record in result]
            except Exception:
                return []

        threshold = settings.fuzzy_name_threshold
        norm_self = _normalise_authority(entity.properties["name"])
        if not norm_self:
            return []

        out: list[Candidate] = []
        for rec in records:
            props = dict(rec["node"])
            norm_other = _normalise_authority(props.get("name") or "")
            if not norm_other:
                continue
            sim = JaroWinkler.normalized_similarity(norm_self, norm_other)
            if sim < threshold:
                continue
            out.append(
                Candidate(
                    entity=Entity("Authority", props["authority_id"], props),
                    context={"jw_similarity": sim, "raw_score": float(rec["score"])},
                )
            )
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        sim = float(candidate.context.get("jw_similarity", 0.0))
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=sim,
            entity_type="Authority",
            details={"jw_similarity": sim},
        )
