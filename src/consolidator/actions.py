"""Action executors — physically modify the graph in response to a Decision.

Every executor is idempotent. Execution order (enforced by engine): the highest-
confidence rule that produces a non-noop decision wins per candidate pair;
subsequent rules for that pair are skipped.

When settings.auto_merge_enabled is False, any "merge" decision is downgraded
to a "flag" (writes :SAME_AS {reviewed:false} instead of collapsing the nodes).
This is the safety valve for the initial rollout.
"""

from datetime import datetime, timezone

from neo4j import AsyncDriver

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def execute(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    entity: Entity,
    candidate: Candidate,
) -> str:
    """Dispatch to the right executor.

    Returns the decision_type actually applied (may differ from decision.action
    when auto_merge is disabled or a conflict is detected).
    """
    if decision.action == "merge":
        if not settings.auto_merge_enabled:
            await _flag_same_as(driver, database, decision=decision, reviewed=False)
            return "flag"
        await _merge(driver, database, decision=decision, entity=entity, candidate=candidate)
        return "auto_merge"

    if decision.action == "link":
        rel_type = decision.details.get("rel_type", "RELATED_TO")
        await _link(driver, database, decision=decision, rel_type=rel_type)
        return "auto_link"

    if decision.action == "flag":
        conflict = bool(decision.details.get("conflict", False))
        await _flag_same_as(
            driver, database, decision=decision, reviewed=False, conflict=conflict
        )
        return "conflict" if conflict else "flag"

    return "noop"


async def _merge(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    entity: Entity,
    candidate: Candidate,
) -> None:
    """Collapse candidate into entity. Rewrites candidate's edges to entity, then deletes candidate.
    Writes a :MergeEvent audit node. Idempotent: if candidate no longer exists, no-op.
    """
    label = decision.entity_type
    id_key = "gmr_id" if label == "Company" else "authority_id"

    async with driver.session(database=database) as session:
        # Check both exist
        result = await session.run(
            f"""
            MATCH (canonical:{label} {{{id_key}: $canonical_id}})
            MATCH (dup:{label} {{{id_key}: $dup_id}})
            RETURN canonical, dup
            """,
            canonical_id=decision.source_id,
            dup_id=decision.target_id,
        )
        record = await result.single()
        if record is None:
            return  # one or both nodes gone — nothing to merge

        # Successor-LEI merges preserve lineage: append the retired LEI to
        # `canonical.historic_leis` BEFORE the mergeNodes call swallows the
        # duplicate node. Other rules don't need this.
        retired_lei = None
        if decision.rule_name == "successor_lei_match":
            retired_lei = (decision.details or {}).get("retired_lei")

        await session.run(
            f"""
            MATCH (canonical:{label} {{{id_key}: $canonical_id}})
            MATCH (dup:{label} {{{id_key}: $dup_id}})
            FOREACH (lei IN CASE WHEN $retired_lei IS NULL THEN [] ELSE [$retired_lei] END |
              SET canonical.historic_leis = coalesce(canonical.historic_leis, []) + lei
            )
            WITH canonical, dup
            CALL apoc.refactor.mergeNodes([canonical, dup], {{
              properties: "discard",
              mergeRels: true
            }}) YIELD node
            WITH node
            CREATE (e:MergeEvent {{
              canonical_id: $canonical_id,
              merged_id: $dup_id,
              merged_at: $merged_at,
              method: $rule_name,
              entity_type: $entity_type,
              retired_lei: $retired_lei
            }})
            RETURN node
            """,
            canonical_id=decision.source_id,
            dup_id=decision.target_id,
            merged_at=_now(),
            rule_name=decision.rule_name,
            entity_type=label,
            retired_lei=retired_lei,
        )


async def _link(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    rel_type: str,
) -> None:
    label = decision.entity_type
    id_key = "gmr_id" if label == "Company" else "authority_id"
    async with driver.session(database=database) as session:
        await session.run(
            f"""
            MATCH (a:{label} {{{id_key}: $source_id}})
            MATCH (b:{label} {{{id_key}: $target_id}})
            MERGE (a)-[r:`{rel_type}`]->(b)
            SET r.created_by_rule = $rule_name, r.confidence = $confidence, r.created_at = $created_at
            """,
            source_id=decision.source_id,
            target_id=decision.target_id,
            rule_name=decision.rule_name,
            confidence=decision.confidence,
            created_at=_now(),
        )


async def _flag_same_as(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    reviewed: bool,
    conflict: bool = False,
) -> None:
    label = decision.entity_type
    id_key = "gmr_id" if label == "Company" else "authority_id"
    async with driver.session(database=database) as session:
        await session.run(
            f"""
            MATCH (a:{label} {{{id_key}: $source_id}})
            MATCH (b:{label} {{{id_key}: $target_id}})
            MERGE (a)-[r:SAME_AS]->(b)
            SET r.confidence = $confidence,
                r.method = $rule_name,
                r.detected_at = $detected_at,
                r.reviewed = $reviewed,
                r.conflict = $conflict
            """,
            source_id=decision.source_id,
            target_id=decision.target_id,
            confidence=decision.confidence,
            rule_name=decision.rule_name,
            detected_at=_now(),
            reviewed=reviewed,
            conflict=conflict,
        )
