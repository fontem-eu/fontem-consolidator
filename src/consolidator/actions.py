"""Action executors — physically modify the graph in response to a Decision.

Every executor is idempotent. Execution order (enforced by engine): the highest-
confidence rule that produces a non-noop decision wins per candidate pair;
subsequent rules for that pair are skipped.

When settings.auto_merge_enabled is False, any "merge" decision is downgraded
to a "flag" (writes :SAME_AS {reviewed:false} instead of collapsing the nodes).
This is the safety valve for the initial rollout. **Per-rule override**: a
decision whose details carry `force_auto_merge: True` (stamped by the engine
when the firing rule sets `Rule.force_auto_merge = True`) bypasses the global
gate. Reserved for deterministic identifier matches — see `rules/base.py`.

Event-log emission: every match (flag, conflict, merge) ALSO emits an
``AssertSameAs`` event into ``events.entity_events`` so the Virtuoso
sink can project ``owl:sameAs``. The Neo4j-direct write here remains
the source of truth for the rich detection-arrays metadata used by
the review queue; a follow-up will move that projection into the
Neo4j sink and let this module emit-only. See CONSOLIDATOR-TRIGGER.md.
"""

from datetime import datetime, timezone

from neo4j import AsyncDriver

from src.config import settings
from src.consolidator import eventlog
from src.consolidator.rules.base import Candidate, Decision, Entity

_ID_KEY_BY_LABEL: dict[str, str] = {
    "Company": "gmr_id",
    "Authority": "authority_id",
    "Contract": "ted_notice_id",
}

# IRI scheme matches the producers (gmr-virtuoso-sink, gmr-neo4j-sink).
# Used to build the AssertSameAs event payload below.
_IRI_LABEL_BY_TYPE: dict[str, str] = {
    "Company":   "Company",
    "Authority": "Authority",
    "Contract":  "Contract",
}


def _id_key(label: str) -> str:
    return _ID_KEY_BY_LABEL.get(label, "authority_id")


def _entity_iri(entity_type: str, entity_id: str) -> str:
    label = _IRI_LABEL_BY_TYPE.get(entity_type, entity_type)
    return f"http://data.fontem.eu/id/{label}/{entity_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()




# Six kwargs: the five-arg executor dispatch contract plus the optional
# emit collector. Bundling them into an object would hide the contract
# every action handler is written against.
async def execute(  # pylint: disable=too-many-arguments
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    entity: Entity,
    candidate: Candidate,
    collect: list[dict] | None = None,
) -> str:
    """Dispatch to the right executor.

    ``collect`` batches the AssertSameAs emits: when a list is passed the
    event is appended rather than written, and the caller emits the whole
    run in one transaction (see eventlog.emit_assert_same_as_many). One
    transaction per run instead of per decision — the difference between
    14.5 and ~2.5 consolidations/sec in prod. Omit it and each event is
    written immediately, which is what the sweeper and any other caller
    outside a retryable request should do.

    Returns the decision_type actually applied (may differ from decision.action
    when auto_merge is disabled or a conflict is detected).
    """
    if decision.action == "merge":
        # The per-rule force_auto_merge stamp lets deterministic rules
        # (exact LEI/CIK/VAT/authority-id, SAME_AS cluster collapse,
        # GLEIF successor) merge even when the global gate is OFF.
        # See rules/base.py:Rule.force_auto_merge.
        forced = bool(decision.details.get("force_auto_merge"))
        if not settings.auto_merge_enabled and not forced:
            await _flag_same_as(driver, database, decision=decision, reviewed=False)
            await _emit_same_as_event(decision, collect)
            return "flag"
        await _merge(driver, database, decision=decision, entity=entity, candidate=candidate)
        await _emit_same_as_event(decision, collect)
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
        await _emit_same_as_event(decision, collect)
        return "conflict" if conflict else "flag"

    if decision.action == "enrich":
        await _enrich(driver, database, decision=decision)
        return "enrich"

    return "noop"


async def _emit_same_as_event(
    decision: Decision, collect: list[dict] | None = None,
) -> None:
    """Emit an AssertSameAs event so Virtuoso (and replay-from-zero of
    Neo4j) see the equivalence.

    With ``collect`` the event is queued for the caller to write as one
    batch at the end of the run; without it, it is written immediately.
    Either way failures are absorbed in the eventlog shim — the Neo4j
    write above is the immediate source of truth and a flaky event store
    must not abort consolidation."""
    a_iri = _entity_iri(decision.entity_type, decision.source_id)
    b_iri = _entity_iri(decision.entity_type, decision.target_id)
    if collect is not None:
        collect.append({
            "a_iri": a_iri,
            "b_iri": b_iri,
            "confidence": float(decision.confidence),
            "method": decision.rule_name,
            "rule": decision.rule_name,
            "domain": decision.entity_type.lower(),
        })
        return
    await eventlog.emit_assert_same_as(
        a_iri=a_iri,
        b_iri=b_iri,
        confidence=float(decision.confidence),
        method=decision.rule_name,
        rule=decision.rule_name,
        domain=decision.entity_type.lower(),
    )


# `entity` / `candidate` kwargs are part of the executor dispatch
# uniform contract (every action handler takes the same shape, even
# when the merge path drives everything off `decision`). Keeping them
# in the signature matters for the engine's typed call site.
async def _merge(  # pylint: disable=unused-argument
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
    id_key = _id_key(label)

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
              // produceSelfRel:false — without it, merging two nodes turns the
              // SAME_AS edge BETWEEN them into a self-loop on the survivor.
              // Verified against prod APOC: identical merge yields 1 self-loop
              // with the default and 0 with this flag. 571 such edges had
              // accumulated, which is what refs.sameas_no_selfloop was failing
              // on — a node is not a duplicate of itself, and the edge asserts
              // it is.
            CALL apoc.refactor.mergeNodes([canonical, dup], {{
              properties: "discard",
              mergeRels: true,
              produceSelfRel: false
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

        # NOTE: the CLIENT_OF / SUPPLIER_OF trade-summary layer was
        # retired with the fontem-api materialiser (#222) — the graph
        # explorer traverses AWARDED / AWARDED_TO directly, so a merge
        # needs no summary-edge rebuild anymore. The old rebuild here
        # was quietly re-seeding a dead cache after every merge.


async def _link(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    rel_type: str,
) -> None:
    label = decision.entity_type
    id_key = _id_key(label)
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
    """Append a detection to the :SAME_AS edge for human review.

    Schema (Neo4j relationships only allow primitives + arrays of
    primitives, so the per-rule detection list is stored as three
    parallel arrays — readers zip them):
      r.detection_rules       : list[str]   ordered, no duplicates
      r.detection_confidences : list[float] aligned to detection_rules
      r.detection_dates       : list[str]   aligned (ISO timestamps)
      r.confidence            : highest of detection_confidences (summary)
      r.method                : rule_name at the max-confidence index
      r.detected_at           : detected_at at the max-confidence index
      r.reviewed              : sticky-once-true (human via /decide)
      r.conflict              : sticky-once-true (any rule reporting it)
    """
    label = decision.entity_type
    id_key = _id_key(label)
    async with driver.session(database=database) as session:
        await session.run(
            f"""
            MATCH (a:{label} {{{id_key}: $source_id}})
            MATCH (b:{label} {{{id_key}: $target_id}})
            MERGE (a)-[r:SAME_AS]->(b)
            // Indices of existing entries to KEEP (those that aren't
            // for the rule firing now — that one's about to be
            // replaced/appended).
            WITH r,
              [i IN range(0, size(coalesce(r.detection_rules, [])) - 1)
                 WHERE coalesce(r.detection_rules, [])[i] <> $rule_name
              ] AS keep
            SET r.detection_rules =
                  [i IN keep | r.detection_rules[i]] + [$rule_name],
                r.detection_confidences =
                  [i IN keep | r.detection_confidences[i]] + [$confidence],
                r.detection_dates =
                  [i IN keep | r.detection_dates[i]] + [$detected_at],
                r.reviewed = (
                  CASE WHEN coalesce(r.reviewed, false) THEN true
                       ELSE $reviewed END
                ),
                r.conflict = (coalesce(r.conflict, false) OR $conflict)
            // Recompute summary fields from the full (post-update)
            // confidence list. The reduce() walks parallel arrays to
            // find the max-confidence index.
            WITH r,
              reduce(best = {{idx: 0, c: -1.0}},
                     i IN range(0, size(r.detection_confidences) - 1) |
                CASE WHEN r.detection_confidences[i] > best.c
                     THEN {{idx: i, c: r.detection_confidences[i]}}
                     ELSE best END
              ).idx AS top_i
            SET r.confidence  = r.detection_confidences[top_i],
                r.method      = r.detection_rules[top_i],
                r.detected_at = r.detection_dates[top_i]
            """,
            source_id=decision.source_id,
            target_id=decision.target_id,
            confidence=decision.confidence,
            rule_name=decision.rule_name,
            detected_at=_now(),
            reviewed=reviewed,
            conflict=conflict,
        )


async def _enrich(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
) -> None:
    """Write translation + (optional) embedding properties back to the node.

    Rules pick the field prefix via ``details["field"]`` — "name" for
    Authority, "title" for Contract — and the executor writes
    ``{field}_<lang>``, ``{field}_embedding``, ``{field}_embedding_encoder``,
    ``{field}_embedding_dim``, ``{field}_lang``. Defaults to "name" so
    older decisions without the field key stay valid.

    When an embedding is present, an encoder identity MUST also be
    present. Un-versioned embeddings would be un-comparable silently
    down the line — we refuse to write them rather than poison the cache.
    """
    label = decision.entity_type
    id_key = _id_key(label)
    field = decision.details.get("field") or "name"
    translations = decision.details.get("translations") or {}
    embedding = decision.details.get("embedding")
    embedding_encoder = decision.details.get("embedding_encoder")
    source_lang = decision.details.get("source_lang")

    if embedding is not None and not embedding_encoder:
        raise ValueError(
            "_enrich: embedding present but embedding_encoder missing; "
            "refusing to write an un-versioned vector",
        )

    props: dict = {}
    for lang, text in translations.items():
        if isinstance(lang, str) and isinstance(text, str) and lang.isalpha():
            props[f"{field}_{lang.lower()}"] = text
    if embedding is not None:
        props[f"{field}_embedding"] = embedding
        props[f"{field}_embedding_encoder"] = embedding_encoder
        props[f"{field}_embedding_dim"] = len(embedding)
    if source_lang:
        props[f"{field}_lang"] = source_lang
    if not props:
        return

    async with driver.session(database=database) as session:
        await session.run(
            f"""
            MATCH (n:{label} {{{id_key}: $id}})
            SET n += $props,
                n.multilingual_updated_at = $now
            """,
            id=decision.source_id,
            props=props,
            # Native datetime → stored as Neo4j DateTime (not ISO string), so
            # `WHERE a.multilingual_updated_at > datetime(...)` filters work.
            now=datetime.now(timezone.utc),
        )
