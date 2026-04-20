"""The consolidation engine: runs the rule pipeline for an entity and persists every outcome."""

from dataclasses import dataclass

from loguru import logger
from neo4j import AsyncDriver
from prometheus_client import Counter

from src.config import settings
from src.consolidator import actions, audit, entities
from src.consolidator.rules.base import Decision
from src.consolidator.rules.registry import list_rules

RULE_FIRES = Counter(
    "gmr_consolidator_rule_fires_total",
    "Rule fires, grouped by rule + outcome + entity type",
    ["rule", "outcome", "entity_type"],
)


@dataclass
class ConsolidationResult:
    run_id: str
    entity_type: str
    entity_id: str
    decisions: list[dict]
    rules_fired: int


async def consolidate(
    driver: AsyncDriver,
    database: str,
    *,
    entity_type: str,
    entity_id: str,
    triggered_by: str = "api",
) -> ConsolidationResult:
    """Run the rule pipeline for an entity. Always records a :ConsolidationRun with outcomes."""

    entity = await entities.load(driver, database, entity_type=entity_type, entity_id=entity_id)

    # Even if the entity is gone we still record a run so history is complete.
    placeholder = entity or entities.Entity(
        entity_type=entity_type, id=entity_id, properties={}
    )  # type: ignore[attr-defined]

    run_id = await audit.start_run(
        driver, database, entity=placeholder, triggered_by=triggered_by
    )

    if entity is None:
        logger.warning(
            "consolidator: entity not found {entity_type}={entity_id}",
            entity_type=entity_type,
            entity_id=entity_id,
        )
        await audit.end_run(
            driver, database, run_id=run_id, rules_fired=0, decisions=0, outcome="not_found"
        )
        return ConsolidationResult(
            run_id=run_id, entity_type=entity_type, entity_id=entity_id, decisions=[], rules_fired=0
        )

    decisions_recorded: list[dict] = []
    rules_fired = 0
    handled_targets: set[str] = set()

    for rule in list_rules():
        if entity_type not in rule.entity_types:
            continue
        try:
            applies = await rule.applies(entity)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("rule {name} applies() raised", name=rule.name)
            continue
        if not applies:
            continue

        candidates = await rule.find_candidates(entity)
        if not candidates:
            continue

        rules_fired += 1
        for candidate in candidates:
            if candidate.entity.id == entity.id:
                continue  # never self-match
            if candidate.entity.id in handled_targets:
                continue  # already had a higher-confidence rule fire

            decision: Decision = await rule.resolve(entity, candidate)

            outcome = await actions.execute(
                driver, database, decision=decision, entity=entity, candidate=candidate
            )
            await audit.record_decision(
                driver,
                database,
                run_id=run_id,
                decision=decision,
                decision_type=outcome,
                candidate=candidate,
            )
            RULE_FIRES.labels(rule=rule.name, outcome=outcome, entity_type=entity_type).inc()
            decisions_recorded.append(
                {
                    "rule_name": rule.name,
                    "action": decision.action,
                    "outcome": outcome,
                    "target_id": decision.target_id,
                    "confidence": decision.confidence,
                }
            )
            if outcome in ("auto_merge", "auto_link"):
                handled_targets.add(candidate.entity.id)

    summary_outcome = _summarize(decisions_recorded)
    await audit.end_run(
        driver,
        database,
        run_id=run_id,
        rules_fired=rules_fired,
        decisions=len(decisions_recorded),
        outcome=summary_outcome,
    )

    logger.info(
        "consolidator: run {run_id} {entity_type}/{entity_id} → {outcome} "
        "({rules} rules, {decisions} decisions, auto_merge={auto})",
        run_id=run_id,
        entity_type=entity_type,
        entity_id=entity_id,
        outcome=summary_outcome,
        rules=rules_fired,
        decisions=len(decisions_recorded),
        auto=settings.auto_merge_enabled,
    )

    return ConsolidationResult(
        run_id=run_id,
        entity_type=entity_type,
        entity_id=entity_id,
        decisions=decisions_recorded,
        rules_fired=rules_fired,
    )


def _summarize(decisions: list[dict]) -> str:
    if not decisions:
        return "no_match"
    if any(d["outcome"] == "auto_merge" for d in decisions):
        return "merged"
    if any(d["outcome"] == "auto_link" for d in decisions):
        return "linked"
    if any(d["outcome"] == "conflict" for d in decisions):
        return "conflict_flagged"
    if any(d["outcome"] == "flag" for d in decisions):
        return "flagged"
    return "noop"
