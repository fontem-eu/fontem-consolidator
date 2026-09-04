"""The consolidation engine: runs the rule pipeline for an entity and persists every outcome."""

from dataclasses import dataclass
from typing import Literal

from loguru import logger
from neo4j import AsyncDriver
from prometheus_client import Counter

from src.config import settings
from src.consolidator import actions, audit, entities, eventlog
from src.consolidator.rules.base import Decision
from src.consolidator.rules.registry import list_rules

# `match_only` skips rules with action="enrich" — useful for the dedup
# sweep where translation enrichment dominates wall-time but is
# orthogonal to matching. `enrich_only` is the inverse, for the
# translation-backfill sweep.
ConsolidateMode = Literal["all", "match_only", "enrich_only"]

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


# The pipeline orchestrator legitimately walks every rule + applies
# per-rule promotion/conflict gates + writes audit + emits metrics.
# Splitting it makes the control flow harder to follow than keeping
# it as one linear function; the kwargs map 1:1 to the public API
# surface (consolidator dispatch route + trigger consumer).
async def _finish_run(  # pylint: disable=too-many-arguments
    driver: AsyncDriver,
    database: str,
    *,
    run_id: str,
    rules_fired: int,
    decisions_recorded: list[dict],
    pending_events: list[dict],
) -> str:
    """Close out a run: flush its events, then record the audit row.

    The flush happens here — before the run returns — so a failure is
    still inside the HTTP request the trigger is waiting on. That is what
    makes batching safe: a crash before this point fails the dispatch, the
    trigger does not advance its offset, and the whole consolidation is
    redelivered and redone.
    """
    await _flush_pending_events(run_id, pending_events)
    summary_outcome = _summarize(decisions_recorded)
    await audit.end_run(
        driver,
        database,
        run_id=run_id,
        rules_fired=rules_fired,
        decisions=len(decisions_recorded),
        outcome=summary_outcome,
    )
    return summary_outcome


async def _flush_pending_events(run_id: str, pending: list[dict]) -> None:
    """Write the run's AssertSameAs events as one transaction.

    Called before the run returns, so a failure here is still inside the
    HTTP request the trigger is waiting on — which is what makes the batch
    safe: a crash before this point fails the dispatch, the trigger does
    not advance its offset, and the whole consolidation is redelivered and
    redone. See eventlog.emit_assert_same_as_many.
    """
    emitted = await eventlog.emit_assert_same_as_many(pending)
    if pending and emitted != len(pending):
        logger.warning(
            "consolidator: run {run} emitted {got}/{want} AssertSameAs events",
            run=run_id, got=emitted, want=len(pending),
        )


async def consolidate(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    driver: AsyncDriver,
    database: str,
    *,
    entity_type: str,
    entity_id: str,
    triggered_by: str = "api",
    exclude_rule_prefix: str | None = None,
    translation_backend: str | None = None,
    mode: ConsolidateMode = "all",
) -> ConsolidationResult:
    """Run the rule pipeline for an entity. Always records a :ConsolidationRun with outcomes.

    exclude_rule_prefix: optional rule-name prefix to skip (e.g. "gds_" for
    fast bulk scans — GDS rules reproject the whole subgraph per call).

    translation_backend: optional per-request override for the gmr-linguistics
    translation backend (e.g. "mistral", "nllb-local"). Threaded into each
    enrichment rule's candidate context; `None` falls back to the service
    default configured on the consolidator pod.

    mode: filters which rules run by their action.
      "all" (default) — every applicable rule runs.
      "match_only"    — skip rules with action="enrich" (e.g. translation
                        enrichment), keep matching/dedup rules. Used by the
                        dedup sweep so it isn't blocked on linguistics RTTs.
      "enrich_only"   — only run rules with action="enrich". Used by the
                        translation-backfill sweep.
    """

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


    # AssertSameAs events for this run, written as one transaction at the
    # end rather than one per decision. See eventlog.emit_assert_same_as_many
    # for why that is safe: the trigger holds its offset until this whole
    # request returns, so a crash before the flush is redelivered and redone.
    # Both accumulate for the lifetime of this run and are consumed
    # together at the end: one is audited, the other emitted.
    decisions_recorded: list[dict] = []
    pending_events: list[dict] = []
    rules_fired = 0
    handled_targets: set[str] = set()

    for rule in list_rules():
        if exclude_rule_prefix and rule.name.startswith(exclude_rule_prefix):
            continue
        if mode == "match_only" and rule.action == "enrich":
            continue
        if mode == "enrich_only" and rule.action != "enrich":
            continue
        if entity_type not in rule.entity_types:
            continue
        try:
            applies = await rule.applies(entity)
        # A buggy rule must not abort the whole pipeline — log + skip.
        except Exception:  # pragma: no cover  # pylint: disable=broad-exception-caught
            logger.exception("rule {name} applies() raised", name=rule.name)
            continue
        if not applies:
            continue

        candidates = await rule.find_candidates(entity)
        if not candidates:
            continue

        if translation_backend and rule.action == "enrich":
            for c in candidates:
                c.context["translation_backend_override"] = translation_backend

        rules_fired += 1
        for candidate in candidates:
            is_self = candidate.entity.id == entity.id
            # Per-entity enrichment rules legitimately target the entity
            # itself (e.g. translation enrichment writes properties back).
            # Other rules must never self-match.
            if is_self and rule.action != "enrich":
                continue
            if not is_self and candidate.entity.id in handled_targets:
                continue  # already had a higher-confidence rule fire

            decision: Decision = await rule.resolve(entity, candidate)

            # Threshold-based promotion: a high-confidence flag with no
            # detected conflict is treated as an auto-merge when the
            # rule has set an auto_merge_threshold. Calibrated per-rule
            # from the canary sweeps; conflict-flagged pairs (mismatched
            # LEI/VAT/etc.) always remain in the human queue regardless
            # of confidence. The actions._merge gate on
            # settings.auto_merge_enabled still applies.
            if (
                decision.action == "flag"
                and rule.auto_merge_threshold is not None
                and decision.confidence >= rule.auto_merge_threshold
                and not decision.details.get("conflict", False)
            ):
                decision = Decision(
                    rule_name=decision.rule_name,
                    action="merge",
                    source_id=decision.source_id,
                    target_id=decision.target_id,
                    confidence=decision.confidence,
                    entity_type=decision.entity_type,
                    details={
                        **decision.details,
                        "auto_merged_above_threshold": rule.auto_merge_threshold,
                    },
                )

            # Stamp the rule's `force_auto_merge` opt-in into the
            # decision so actions.execute can bypass the global
            # `auto_merge_enabled` gate for deterministic identifier
            # matches. Conflict-flagged decisions never get the
            # stamp (the conflict branch in conflict.py emits
            # action=flag with details.conflict=True, which keeps
            # them out of this code path anyway).
            if (
                rule.force_auto_merge
                and decision.action == "merge"
                and not decision.details.get("conflict", False)
            ):
                decision = Decision(
                    rule_name=decision.rule_name,
                    action=decision.action,
                    source_id=decision.source_id,
                    target_id=decision.target_id,
                    confidence=decision.confidence,
                    entity_type=decision.entity_type,
                    details={**decision.details, "force_auto_merge": True},
                )

            outcome = await actions.execute(
                driver, database, decision=decision, collect=pending_events,
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
            # Short-circuit only on graph-mutating outcomes:
            # `auto_merge` collapses the target node, `auto_link` writes a
            # named relationship — running another rule on the same target
            # afterwards is undefined.
            #
            # `flag` and `conflict` write a :SAME_AS_CANDIDATE and APPEND
            # a detection to it, so multiple rules firing on the same pair
            # leave the reviewer richer evidence — exact-name + cosine +
            # fuzzy can all be recorded for one proposal. They assert
            # nothing and emit nothing; see actions.py.
            if outcome in ("auto_assert", "auto_link"):
                handled_targets.add(candidate.entity.id)
            # Enrichment never participates in the short-circuit: it's
            # orthogonal to matching — a merged pair can still want
            # translations filled in.

    summary_outcome = await _finish_run(
        driver, database, run_id=run_id, rules_fired=rules_fired,
        decisions_recorded=decisions_recorded, pending_events=pending_events,
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


# Each branch maps one outcome to one summary label — flattening into
# a dict would just hide the priority ordering (merge > link > conflict
# > flag > enrich > noop) that this if-ladder makes explicit.
def _summarize(decisions: list[dict]) -> str:  # pylint: disable=too-many-return-statements
    if not decisions:
        return "no_match"
    if any(d["outcome"] == "auto_assert" for d in decisions):
        return "asserted"
    if any(d["outcome"] == "auto_link" for d in decisions):
        return "linked"
    if any(d["outcome"] == "conflict" for d in decisions):
        return "conflict_flagged"
    if any(d["outcome"] == "flag" for d in decisions):
        return "flagged"
    if any(d["outcome"] == "enrich" for d in decisions):
        return "enriched"
    return "noop"
