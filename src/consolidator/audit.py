from datetime import datetime, timezone
from uuid import uuid4

from neo4j import AsyncDriver

from src.consolidator.rules.base import Candidate, Decision, Entity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Audit writer kwargs mirror the :ConsolidationRun / :DecisionLog node
# shape; bundling them into a struct would just hide the columns.
async def start_run(  # pylint: disable=too-many-arguments
    driver: AsyncDriver,
    database: str,
    *,
    entity: Entity,
    triggered_by: str,
) -> str:
    run_id = str(uuid4())
    async with driver.session(database=database) as session:
        await session.run(
            """
            CREATE (r:ConsolidationRun {
              run_id: $run_id,
              started_at: $started_at,
              triggered_by: $triggered_by,
              entity_type: $entity_type,
              entity_id: $entity_id,
              rules_fired_count: 0,
              decisions_count: 0
            })
            """,
            run_id=run_id,
            started_at=_now(),
            triggered_by=triggered_by,
            entity_type=entity.entity_type,
            entity_id=entity.id,
        )
    return run_id


# See start_run() — six kwargs map 1:1 to the :ConsolidationRun columns
# being patched at run end (rules_fired, decisions, outcome).
async def end_run(  # pylint: disable=too-many-arguments
    driver: AsyncDriver,
    database: str,
    *,
    run_id: str,
    rules_fired: int,
    decisions: int,
    outcome: str,
) -> None:
    async with driver.session(database=database) as session:
        await session.run(
            """
            MATCH (r:ConsolidationRun {run_id: $run_id})
            SET r.ended_at = $ended_at,
                r.rules_fired_count = $rules_fired,
                r.decisions_count = $decisions,
                r.outcome = $outcome
            """,
            run_id=run_id,
            ended_at=_now(),
            rules_fired=rules_fired,
            decisions=decisions,
            outcome=outcome,
        )


# See start_run() — kwargs mirror :RuleApplication + :DecisionLog
# columns one-for-one and are the audit-log public contract.
async def record_decision(  # pylint: disable=too-many-arguments
    driver: AsyncDriver,
    database: str,
    *,
    run_id: str,
    decision: Decision,
    decision_type: str,
    candidate: Candidate,
    reviewer: str | None = None,
    review_note: str | None = None,
) -> str:
    """Record a :RuleApplication + :DecisionLog chain.

    decision_type is the concrete outcome after the action executor runs:
      "auto_assert" | "auto_link" | "flag" | "conflict" | "noop"
      | "manual_merge" | "manual_reject" | "manual_keep_related"
    """
    decision_id = str(uuid4())
    application_id = str(uuid4())
    async with driver.session(database=database) as session:
        await session.run(
            """
            MATCH (r:ConsolidationRun {run_id: $run_id})
            CREATE (ra:RuleApplication {
              application_id: $application_id,
              rule_name: $rule_name,
              confidence: $confidence,
              action: $action,
              applied_at: $applied_at,
              candidate_id: $candidate_id,
              details: $details
            })
            CREATE (r)-[:APPLIED]->(ra)
            CREATE (dl:DecisionLog {
              decision_id: $decision_id,
              decided_at: $decided_at,
              decision_type: $decision_type,
              rule_name: $rule_name,
              confidence: $confidence,
              source_id: $source_id,
              target_id: $target_id,
              entity_type: $entity_type,
              reviewer: $reviewer,
              review_note: $review_note
            })
            CREATE (ra)-[:PRODUCED]->(dl)
            """,
            run_id=run_id,
            application_id=application_id,
            rule_name=decision.rule_name,
            confidence=decision.confidence,
            action=decision.action,
            applied_at=_now(),
            candidate_id=candidate.entity.id,
            details=str(decision.details),
            decision_id=decision_id,
            decided_at=_now(),
            decision_type=decision_type,
            source_id=decision.source_id,
            target_id=decision.target_id,
            entity_type=decision.entity_type,
            reviewer=reviewer,
            review_note=review_note,
        )
    return decision_id
