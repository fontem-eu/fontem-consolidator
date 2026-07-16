"""EmbeddingCosineSameAuthority — flag same-entity-in-different-language pairs.

Uses the name_embedding stored on every :Authority (cosine over whatever
multilingual encoder the linguistics service produced — Mistral,
LaBSE, MiniLM) to find cross-lingual / cross-country duplicates that the
string-based rules miss (e.g. ``Ministero della Difesa`` ↔ ``Ministry of
Defence``).

Never auto-merges. Always flags for review via the standard
``_flag_same_as`` executor — human decides because embedding cosine is
semantic, not deterministic.

Safety invariants:
- **Homogeneity is enforced query-side**: the Cypher WHERE gate
  ``node.name_embedding_encoder = $enc`` guarantees every compared pair
  shares the same encoder. Cross-encoder cosines are meaningless (different
  vector spaces), and this DB-side filter is the guard. There is no
  app-level allowlist; any encoder produced by the linguistics service
  is legitimate for comparison against its own siblings.
- Filters out pairs already reviewed by a human (``reviewed=true`` on
  the existing :SAME_AS edge).
- Uses Neo4j's native vector index for O(log n) k-NN lookup; without
  the index this rule would run a full 61k × <dim> dot product per entity.

Calibration note:
The auto_merge_threshold below (0.98) was calibrated on a 500-authority
LaBSE canary. Different encoders have different cosine distributions;
if enrichment starts producing embeddings under a new encoder_id you
should re-canary before trusting the auto_merge tier on those pairs.
"""
from __future__ import annotations

from loguru import logger
from rapidfuzz.distance import JaroWinkler

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class EmbeddingCosineSameAuthority(Rule):
    name = "embedding_cosine_authority"
    description = (
        "Flag :Authority pairs whose name embeddings are cosine-similar "
        "above threshold. Targets cross-lingual / cross-country duplicates "
        "(same entity named in different languages) that string-similarity "
        "rules miss. Never auto-merges — writes :SAME_AS {reviewed:false} "
        "for human decision. Compares pairs only within the same encoder "
        "family (Cypher-enforced), so mixing encoders in the graph is safe."
    )
    entity_types = {"Authority"}
    # Sits between ExactNameAnyCountryAuthority (0.90) and the GDS rules
    # (0.80). Rule-level confidence is the tier; the per-decision
    # confidence is the actual cosine score, stored on the :SAME_AS edge.
    confidence = 0.87
    action = "flag"
    # Calibrated on the 500-authority canary at JW=0.45: at cosine ≥
    # 0.98 the matches are exclusively EU-body cross-country variants
    # (DG COMM / FPI / INTPA per delegation) — zero false positives
    # observed. Below 0.98, results trail off into role-only twins
    # (national libraries, central banks, music academies). Auto-merge
    # at the top tier; flag the rest for human review.
    auto_merge_threshold = 0.98

    async def applies(self, entity: Entity) -> bool:
        if not settings.embedding_cosine_enabled:
            return False
        vec = entity.properties.get("name_embedding")
        enc = entity.properties.get("name_embedding_encoder")
        # Require the entity to carry both a vector and an encoder-id, but
        # don't gate on which encoder it is — the Cypher WHERE clause in
        # find_candidates() enforces "same encoder on both sides", which
        # is the property that makes the cosine meaningful. Refusing here
        # on a non-allowlisted encoder is what silently broke the rule
        # for 52 days when the linguistics service was returning
        # mistral-embed encoder-ids and this gate insisted on LaBSE.
        if not isinstance(vec, list) or not vec:
            return False
        if not isinstance(enc, str) or not enc:
            return False
        return True

    # Vector-index lookup + Cypher filter stack + Python-side
    # Jaro-Winkler gate all pivot off the same per-call settings —
    # the locals stay readable inline rather than spread across
    # helper signatures.
    async def find_candidates(self, entity: Entity) -> list[Candidate]:  # pylint: disable=too-many-locals
        # Imported lazily so unit tests that patch `get_driver` don't see
        # a module-level import resolving before their monkeypatch.
        from src.consolidator.neo4j.client import get_driver  # pylint: disable=import-outside-toplevel

        driver = await get_driver()
        vec = entity.properties["name_embedding"]
        enc = entity.properties["name_embedding_encoder"]
        self_name = entity.properties.get("name") or ""
        self_country = entity.properties.get("country") or ""
        top_k = settings.embedding_cosine_top_k
        threshold = settings.embedding_cosine_threshold
        jw_min = settings.embedding_cosine_jaro_winkler_min
        cross_country_only = settings.embedding_cosine_cross_country_only

        # The filter stack:
        #   1. self_id ≠ candidate (trivial, in DB)
        #   2. encoder_id match (string equality, in DB)
        #   3. cosine ≥ threshold (already applied by the vector index)
        #   4. cross-country gate (in DB, when enabled)
        #   5. drop pairs already reviewed by a human (OPTIONAL MATCH)
        #   6. minimum Jaro-Winkler on raw names — applied in Python via
        #      rapidfuzz, same library as FuzzyNameSameCountryAuthority.
        #      APOC doesn't ship a jaroWinkler function (only Levenshtein
        #      distance), and reusing rapidfuzz keeps the metric
        #      consistent with the existing fuzzy rule.
        query = """
        CALL db.index.vector.queryNodes(
            'authority_name_embedding_idx', $k, $vec
        ) YIELD node, score
        WHERE node.authority_id <> $self_id
          AND node.name_embedding_encoder = $enc
          AND score >= $threshold
          AND ($cross_country_only = false
               OR coalesce(node.country, '') <> coalesce($self_country, ''))
        OPTIONAL MATCH (s:Authority {authority_id: $self_id})-[r:SAME_AS]->(node)
        WITH node, score, r
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
                self_country=self_country,
                cross_country_only=cross_country_only,
            )
            records = [record async for record in result]

        # Post-filter on Jaro-Winkler against the source name. Done in
        # Python because (a) APOC has no jaroWinkler function and (b)
        # rapidfuzz is already in the consolidator's deps + used by the
        # fuzzy rule. The set we filter is at most top_k+1 rows so the
        # cost is negligible.
        self_name_lc = self_name.lower()
        if jw_min > 0.0:
            kept = []
            for rec in records:
                name = (dict(rec["n"]).get("name") or "").lower()
                if JaroWinkler.normalized_similarity(self_name_lc, name) >= jw_min:
                    kept.append(rec)
            records = kept

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
