"""Fuzzy name rule — flag (:SAME_AS) candidates within the same country when
the normalised names are Jaro-Winkler-similar above the threshold.

Pipeline:
  1. Use the Company.name fulltext index to fetch candidate :Company nodes
     in the same country (cheap retrieval — we still cap to 20).
  2. Normalise both names: upper-case, drop legal-form suffixes (AB, GmbH,
     SAS, LLP, LTD, ...), drop punctuation, collapse whitespace.
  3. Compute Jaro-Winkler similarity on the normalised pair.
  4. Skip if hard identifiers disagree (different valid LEI / CIK / VAT) —
     those pairs are definitely different entities, no point flagging.
  5. Emit :SAME_AS at the actual similarity score (not clamped) when
     sim >= settings.fuzzy_name_threshold.
"""

import re

from rapidfuzz.distance import JaroWinkler

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule
from src.consolidator.rules.conflict import find_conflict

_LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')

# Legal-form tokens to strip before similarity (case-insensitive word-boundary).
# Keep this conservative — stripping too much causes different entities to look
# identical ("Acme AB" vs "Acme Ltd" should still compare as "ACME" = "ACME" → 1.0
# which is the correct call since hard-id conflict detection handles the
# "actually different legal entity" case).
# Alternation order matters — longer patterns FIRST so "S.A.R.L." wins over
# the shorter "S.A" prefix match.
_LEGAL_SUFFIX = re.compile(
    r"\b("
    r"SARL|S\.?A\.?R\.?L\.?|"
    r"S\.?A\.?S\.?|SAS|"
    r"SPRL|SRL|SPA|"
    r"GMBH|LTDA|LLP|LLC|LTD|"
    r"OYJ|ASA|APS|PLC|PTE|PTY|"
    r"S\.?A\.?|"
    r"AB|AG|AS|BV|INC|KFT|KG|KK|NV|OY|SE|SL|SLU|SP|UG"
    r")\.?\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def _lucene_sanitize(name: str) -> str:
    return _LUCENE_SPECIAL.sub(" ", name).strip()


def _normalise(name: str) -> str:
    s = name.upper()
    s = _LEGAL_SUFFIX.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s


class FuzzyNameSameCountry(Rule):
    name = "fuzzy_name_same_country"
    description = (
        "Full-text candidate fetch + Jaro-Winkler similarity on normalised "
        "names (same country). Flags :SAME_AS above "
        f"{settings.fuzzy_name_threshold}. Pairs with disagreeing hard IDs are skipped."
    )
    entity_types = {"Company"}
    confidence = 0.95  # upper bound; emitted confidence is the actual Jaro-Winkler score
    action = "flag"

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name")) and bool(entity.properties.get("country"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        sanitized = _lucene_sanitize(entity.properties["name"])
        if not sanitized:
            return []
        async with driver.session() as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes('company_name_ft', $search_text)
                    YIELD node, score
                    WHERE node.gmr_id <> $self_id
                      AND node.country = $country
                    RETURN node, score
                    LIMIT 20
                    """,
                    search_text=sanitized,
                    self_id=entity.id,
                    country=entity.properties["country"],
                )
                records = [record async for record in result]
            except Exception as exc:  # index may not exist in ephemeral test DBs
                from loguru import logger

                logger.warning("fuzzy_name: fulltext query failed: {}", exc)
                return []

        threshold = settings.fuzzy_name_threshold
        norm_self = _normalise(entity.properties["name"])
        if not norm_self:
            return []

        out: list[Candidate] = []
        for rec in records:
            props = dict(rec["node"])
            norm_other = _normalise(props.get("name") or "")
            if not norm_other:
                continue
            sim = JaroWinkler.normalized_similarity(norm_self, norm_other)
            if sim < threshold:
                continue
            out.append(
                Candidate(
                    entity=Entity("Company", props["gmr_id"], props),
                    context={"jw_similarity": sim, "raw_score": float(rec["score"])},
                )
            )
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        # If the two nodes disagree on a canonicalisable hard identifier,
        # they are definitely different entities. Skip without writing.
        if find_conflict(entity, candidate) is not None:
            return Decision(
                rule_name=self.name,
                action="noop",
                source_id=entity.id,
                target_id=candidate.entity.id,
                confidence=0.0,
                entity_type="Company",
                details={
                    "skipped": "hard_id_conflict",
                    "jw_similarity": candidate.context.get("jw_similarity"),
                },
            )
        sim = float(candidate.context.get("jw_similarity", 0.0))
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=sim,
            entity_type="Company",
            details={"jw_similarity": sim},
        )
