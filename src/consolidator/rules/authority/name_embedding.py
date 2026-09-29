"""AuthorityNameEmbedding — the name_embedding vector a matching rule needs.

EmbeddingCosineSameAuthority finds the same authority named in two
languages ("Ministero della Difesa" / "Ministry of Defence") by comparing
LaBSE vectors of their names. Computing that vector is part of matching,
so it stays here. Translating the name is not: fontem-translator publishes
TranslateAuthorityName events, and this rule used to do both.

Per-entity (the candidate is the entity itself); the engine dispatches
`action == "enrich"` to the executor that writes the vector back. Fails
soft: linguistics unreachable is a noop, and a later run retries.
"""
from __future__ import annotations

from loguru import logger

from src.config import EMBEDDING_BACKEND_DIMS, settings
from src.consolidator.clients.linguistics import (
    LinguisticsClient,
    LinguisticsError,
    LinguisticsUnavailable,
)
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


def needs_embedding(entity: Entity) -> bool:
    """No vector yet, or one from an encoder of another dimension: the
    1024-d mistral-embed vectors made before the move to LaBSE (768-d)
    cannot sit in the index or be compared with LaBSE ones."""
    vec = entity.properties.get("name_embedding")
    want = EMBEDDING_BACKEND_DIMS.get(settings.linguistics_embedding_backend)
    return not isinstance(vec, list) or len(vec) == 0 or (
        want is not None and len(vec) != want)


class AuthorityNameEmbedding(Rule):
    name = "authority_name_embedding"
    description = (
        "Compute the name_embedding vector on :Authority for the "
        "cross-lingual similarity rule. Runs per-entity; never merges or flags."
    )
    entity_types = {"Authority"}
    confidence = 1.0
    action = "enrich"

    async def applies(self, entity: Entity) -> bool:
        return (settings.linguistics_enabled and bool(entity.properties.get("name"))
                and needs_embedding(entity))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        return [Candidate(entity=entity, context={"enrichment": True})]

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        try:
            async with LinguisticsClient(
                base_url=settings.linguistics_url,
                timeout_s=settings.linguistics_timeout_s,
                embedding_backend=settings.linguistics_embedding_backend,
            ) as client:
                embedding, encoder_id = await client.embed(entity.properties["name"])
        except (LinguisticsUnavailable, LinguisticsError) as exc:
            logger.warning("authority_name_embedding: {id} not embedded: {exc}",
                           id=entity.id, exc=exc)
            return Decision(
                rule_name=self.name, action="noop", source_id=entity.id,
                target_id=entity.id, confidence=0.0, entity_type="Authority",
                details={"reason": type(exc).__name__})
        return Decision(
            rule_name=self.name, action="enrich", source_id=entity.id,
            target_id=entity.id, confidence=1.0, entity_type="Authority",
            details={"field": "name", "embedding": embedding,
                     "embedding_encoder": encoder_id},
        )
