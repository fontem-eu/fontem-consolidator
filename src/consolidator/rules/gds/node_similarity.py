"""GDS node-similarity rules.

Project a neighborhood around the anchor entity, run gds.nodeSimilarity in
Jaccard mode, emit candidate pairs above threshold as :SAME_AS flags.

One implementation, two entity-type configurations.
"""

from __future__ import annotations

from src.config import settings
from src.consolidator.neo4j.gds import gds_available, projected_subgraph
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class _GdsNodeSimilarityBase(Rule):
    """Shared find_candidates/resolve. Subclass fills in entity_types + projection config."""

    action = "flag"
    confidence = 0.8  # upper bound; emitted confidence is the Jaccard score

    # subclass config
    anchor_label: str        # "Company" | "Authority"
    neighborhood_labels: tuple[str, ...]
    neighborhood_rels: tuple[str, ...]
    id_key: str              # "gmr_id" | "authority_id"

    async def applies(self, entity: Entity) -> bool:
        return entity.entity_type == self.anchor_label

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver = await get_driver()
        if not await gds_available(driver, "neo4j"):
            return []

        labels = "|".join(self.neighborhood_labels + (self.anchor_label,))
        rels = "|".join(self.neighborhood_rels) or "*"
        node_q = f"""
            MATCH (n)
            WHERE any(lbl IN labels(n) WHERE lbl IN {list(self.neighborhood_labels + (self.anchor_label,))})
            RETURN id(n) AS id, labels(n) AS labels, n.{self.id_key} AS entity_id
        """
        rel_q = f"""
            MATCH (a)-[r:{rels}]-(b)
            WHERE any(lbl IN labels(a) WHERE lbl IN {list(self.neighborhood_labels + (self.anchor_label,))})
              AND any(lbl IN labels(b) WHERE lbl IN {list(self.neighborhood_labels + (self.anchor_label,))})
            RETURN id(a) AS source, id(b) AS target
        """

        out: list[Candidate] = []
        try:
            async with projected_subgraph(
                driver, node_query=node_q, relationship_query=rel_q, database="neo4j"
            ) as graph_name:
                async with driver.session() as session:
                    result = await session.run(
                        f"""
                        CALL gds.nodeSimilarity.stream($graph, {{
                          similarityCutoff: $cutoff,
                          topK: $topK
                        }})
                        YIELD node1, node2, similarity
                        WITH gds.util.asNode(node1) AS a, gds.util.asNode(node2) AS b, similarity
                        WHERE a.{self.id_key} = $self_id
                          AND "{self.anchor_label}" IN labels(b)
                          AND b.{self.id_key} IS NOT NULL
                          AND b.{self.id_key} <> $self_id
                        RETURN b AS candidate, similarity
                        """,
                        graph=graph_name,
                        cutoff=settings.gds_similarity_threshold,
                        topK=settings.gds_top_k,
                        self_id=entity.id,
                    )
                    records = [rec async for rec in result]
            for rec in records:
                props = dict(rec["candidate"])
                score = float(rec["similarity"])
                out.append(
                    Candidate(
                        entity=Entity(self.anchor_label, props[self.id_key], props),
                        context={"jaccard": score},
                    )
                )
        except Exception:
            # GDS projection might fail on a tiny/empty graph (integration tests);
            # behave as noop rather than crash the pipeline.
            return []
        _ = labels  # pyflakes
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        jaccard = float(candidate.context.get("jaccard", 0.0))
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=min(0.95, jaccard),
            entity_type=entity.entity_type,
            details={"jaccard": jaccard, "method": "gds_node_similarity"},
        )


class GdsNodeSimilarityCompany(_GdsNodeSimilarityBase):
    name = "gds_node_similarity_company"
    description = (
        "GDS Jaccard node-similarity over :Company neighborhoods (listings, "
        "financials, contracts). Flags structural look-alikes for review."
    )
    entity_types = {"Company"}
    anchor_label = "Company"
    neighborhood_labels = ("Listing", "FinancialYear", "Contract", "CohesionProject", "NUTSRegion")
    neighborhood_rels = (
        "LISTED_AS",
        "REPORTED",
        "AWARDED_TO",
        "BENEFICIARY_OF",
        "LOCATED_IN",
    )
    id_key = "gmr_id"


class GdsNodeSimilarityAuthority(_GdsNodeSimilarityBase):
    name = "gds_node_similarity_authority"
    description = (
        "GDS Jaccard node-similarity over :Authority award patterns. "
        "Two authorities that award contracts to largely the same companies "
        "are likely duplicates of the same public body."
    )
    entity_types = {"Authority"}
    anchor_label = "Authority"
    neighborhood_labels = ("Contract", "Company", "NUTSRegion")
    neighborhood_rels = ("AWARDED", "AWARDED_TO", "LOCATED_IN")
    id_key = "authority_id"
