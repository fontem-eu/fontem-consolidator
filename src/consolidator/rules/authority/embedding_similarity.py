"""EmbeddingCosineSameAuthority — flag same-entity-in-different-language pairs.

Uses the LaBSE name_embedding stored on every :Authority (768-d, cosine)
to find cross-lingual and cross-country duplicates that the string-based
rules miss (e.g. ``Ministero della Difesa`` ↔ ``Ministry of Defence``).

Never auto-merges. Always flags for review via the standard
``_flag_same_as`` executor — human decides because embedding cosine is
semantic, not deterministic.

Safety invariants:
- Skips entities whose encoder_id isn't on an allowlist (cross-encoder
  comparisons are meaningless, so un-versioned or foreign vectors are
  ignored).
- Filters out pairs already reviewed by a human (``reviewed=true`` on
  the existing :SAME_AS edge).
- Uses Neo4j's native vector index for O(log n) k-NN lookup; without
  the index this rule would run a full 61k × 768 dot product per entity.
"""
from __future__ import annotations

from loguru import logger

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


def _accepted_encoders() -> frozenset[str]:
    raw = settings.embedding_cosine_accepted_encoders or ""
    return frozenset(e.strip() for e in raw.split(",") if e.strip())


class EmbeddingCosineSameAuthority(Rule):
    name = "embedding_cosine_authority"
    description = (
        "Flag :Authority pairs whose LaBSE name embeddings are cosine-"
        "similar above threshold. Targets cross-lingual / cross-country "
        "duplicates (same entity named in different languages) that "
        "string-similarity rules miss. Never auto-merges — writes "
        ":SAME_AS {reviewed:false} for human decision."
    )
    entity_types = {"Authority"}
    # Sits between ExactNameAnyCountryAuthority (0.90) and the GDS rules
    # (0.80). Rule-level confidence is the tier; the per-decision
    # confidence is the actual cosine score, stored on the :SAME_AS edge.
    confidence = 0.87
    action = "flag"

    async def applies(self, entity: Entity) -> bool:
        if not settings.embedding_cosine_enabled:
            return False
        vec = entity.properties.get("name_embedding")
        enc = entity.properties.get("name_embedding_encoder")
        if not isinstance(vec, list) or not vec:
            return False
        if not isinstance(enc, str) or enc not in _accepted_encoders():
            return False
        return True

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        # Imported lazily so unit tests that patch `get_driver` don't see
        # a module-level import resolving before their monkeypatch.
        from src.consolidator.neo4j.client import get_driver  # pylint: disable=import-outside-toplevel

        driver = await get_driver()
        vec = entity.properties["name_embedding"]
        enc = entity.properties["name_embedding_encoder"]
        top_k = settings.embedding_cosine_top_k
        threshold = settings.embedding_cosine_threshold

        query = """
        CALL db.index.vector.queryNodes(
            'authority_name_embedding_idx', $k, $vec
        ) YIELD node, score
        WHERE node.authority_id <> $self_id
          AND node.name_embedding_encoder = $enc
          AND score >= $threshold
        OPTIONAL MATCH (s:Authority {authority_id: $self_id})-[r:SAME_AS]->(node)
        WITH node, score, r
        // Skip pairs a human already reviewed; re-flagging them is
        // noise. New pairs (no edge yet) and unreviewed edges both pass.
        WHERE r IS NULL OR coalesce(r.reviewed, false) = false
        RETURN node AS n, score AS s
        ORDER BY s DESC
        """
        async with driver.session(database=settings.neo4j_database) as session:
            # +1 on k because the nearest neighbour is almost always the
            # query vector itself (identity), which we filter on self_id.
            result = await session.run(
                query, k=top_k + 1, vec=vec,
                self_id=entity.id, enc=enc, threshold=threshold,
            )
            records = [record async for record in result]

        if records:
            logger.debug(
                "embedding_cosine_authority: {n} candidates for {id} "
                "(top score {top:.3f})",
                n=len(records), id=entity.id, top=records[0]["s"],
            )
        return [
            Candidate(
                entity=Entity(
                    entity_type="Authority",
                    id=dict(rec["n"])["authority_id"],
                    properties=dict(rec["n"]),
                ),
                context={"cosine_score": float(rec["s"])},
            )
            for rec in records
        ]

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        cosine = candidate.context["cosine_score"]
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            # Per-decision confidence is the actual cosine — edge
            # consumers see the strength of match, not a flat tier.
            confidence=cosine,
            entity_type="Authority",
            details={
                "method": "embedding_cosine",
                "cosine_score": cosine,
                "encoder_id": entity.properties.get("name_embedding_encoder"),
                "source_country": entity.properties.get("country"),
                "target_country": candidate.entity.properties.get("country"),
            },
        )
