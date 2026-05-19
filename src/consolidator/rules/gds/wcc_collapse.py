"""GDS Weakly-Connected-Components rule: collapse clusters of reviewed :SAME_AS edges.

After humans confirm fuzzy matches by setting `:SAME_AS {reviewed: true}`, the
graph has clusters of equivalent nodes. This rule walks that subgraph, picks a
canonical per cluster, and merges the rest.

Canonical selection order (stable):
  1. Has an LEI (for Company) / has an authority_id (for Authority — always true, so moot)
  2. Oldest `created_at`
  3. Highest node id (deterministic tiebreak)
"""

from __future__ import annotations

from neo4j import AsyncDriver

from src.consolidator.neo4j.gds import gds_available, projected_subgraph
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class _GdsSameAsClusterCollapseBase(Rule):
    action = "merge"
    confidence = 1.0
    # Cluster collapse only fires for entities connected by SAME_AS
    # edges that another rule has already marked `reviewed=true` (the
    # `projected_subgraph` query upstream filters on it). So by the
    # time this rule emits a merge, the equivalence has already been
    # vetted — auto-merge is safe regardless of the global gate.
    force_auto_merge = True

    anchor_label: str
    id_key: str

    async def applies(self, entity: Entity) -> bool:
        return entity.entity_type == self.anchor_label

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        from src.consolidator.neo4j.client import get_driver

        driver: AsyncDriver = await get_driver()
        if not await gds_available(driver, "neo4j"):
            return []

        # Cheap pre-check: only build a GDS projection if THIS
        # entity has at least one reviewed-and-clean SAME_AS edge.
        # Without this, every per-event dispatch projects the full
        # ~3.3M-Company subgraph just to discover the entity isn't
        # in any cluster — measured at ~5s/event.
        #
        # When in_cluster=true we still project the whole label
        # plus reviewed edges (cheap because the WCC-relevant edge
        # set is tiny), but only ~0.01% of entities actually
        # belong to a reviewed cluster, so the average case is
        # an indexed point lookup that returns false.
        async with driver.session() as session:
            res = await session.run(
                f"""
                MATCH (n:{self.anchor_label} {{{self.id_key}: $self_id}})
                OPTIONAL MATCH (n)-[r:SAME_AS]-(:{self.anchor_label})
                WHERE r.reviewed = true AND coalesce(r.conflict, false) = false
                WITH r LIMIT 1
                RETURN r IS NOT NULL AS in_cluster
                """,
                self_id=entity.id,
            )
            row = await res.single()
        if row is None or not row["in_cluster"]:
            return []

        node_q = f"""
            MATCH (n:{self.anchor_label})
            RETURN id(n) AS id, labels(n) AS labels
        """
        rel_q = f"""
            MATCH (a:{self.anchor_label})-[r:SAME_AS]-(b:{self.anchor_label})
            WHERE r.reviewed = true AND coalesce(r.conflict, false) = false
            RETURN id(a) AS source, id(b) AS target
        """

        try:
            async with projected_subgraph(
                driver, node_query=node_q, relationship_query=rel_q, database="neo4j"
            ) as graph_name:
                async with driver.session() as session:
                    # For the entity's component, list siblings and decide canonical.
                    result = await session.run(
                        f"""
                        CALL gds.wcc.stream($graph) YIELD nodeId, componentId
                        WITH gds.util.asNode(nodeId) AS n, componentId
                        WITH componentId,
                             collect({{
                               id: n.{self.id_key},
                               has_lei: n.lei IS NOT NULL,
                               created_at: coalesce(n.created_at, ""),
                               neo_id: id(n)
                             }}) AS members
                        WHERE size(members) > 1 AND any(m IN members WHERE m.id = $self_id)
                        RETURN members
                        """,
                        graph=graph_name,
                        self_id=entity.id,
                    )
                    records = [rec async for rec in result]
        except Exception:
            return []

        if not records:
            return []

        members = records[0]["members"]
        # Deterministic canonical selection
        members_sorted = sorted(
            members,
            key=lambda m: (
                not m["has_lei"],                     # True (has LEI) first
                m["created_at"] or "\uffff",          # oldest first (empty strings sort last)
                m["neo_id"],                          # stable tiebreak
            ),
        )
        canonical = members_sorted[0]
        if canonical["id"] != entity.id:
            # Only emit candidates when we are the canonical; otherwise the canonical
            # run will do the merging. This keeps one run responsible per cluster.
            return []
        candidates = []
        for m in members_sorted[1:]:
            candidates.append(
                Candidate(
                    entity=Entity(self.anchor_label, m["id"], {"neo_id": m["neo_id"]}),
                    context={"component": "wcc_same_as", "canonical_id": canonical["id"]},
                )
            )
        return candidates

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        return Decision(
            rule_name=self.name,
            action="merge",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type=entity.entity_type,
            details={"method": "gds_wcc_same_as_cluster"},
        )


class GdsSameAsClusterCollapseCompany(_GdsSameAsClusterCollapseBase):
    name = "gds_same_as_cluster_collapse_company"
    description = (
        "Walk WCC on reviewed :SAME_AS {reviewed:true} between :Company nodes; "
        "merge each cluster into its canonical (LEI-bearing → oldest → deterministic)."
    )
    entity_types = {"Company"}
    anchor_label = "Company"
    id_key = "gmr_id"


class GdsSameAsClusterCollapseAuthority(_GdsSameAsClusterCollapseBase):
    name = "gds_same_as_cluster_collapse_authority"
    description = "Walk WCC on reviewed :SAME_AS between :Authority; merge to canonical."
    entity_types = {"Authority"}
    anchor_label = "Authority"
    id_key = "authority_id"
