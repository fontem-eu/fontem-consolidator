"""Action executors — record consolidation decisions.

Every executor is idempotent. Execution order (enforced by engine): the highest-
confidence rule that produces a non-noop decision wins per candidate pair;
subsequent rules for that pair are skipped.

Where identity lives
--------------------
Neo4j holds the graph and the review workflow. Virtuoso holds identity.
Nothing asserts an equivalence in both, which is the whole reason for
adopting Virtuoso: ``owl:sameAs`` there is closed transitively and
symmetrically by the store, while a ``:SAME_AS`` edge in Neo4j is just an
edge that nothing follows.

  Neo4j  :SAME_AS_CANDIDATE  a rule proposes these might be the same.
                             ``status`` runs pending -> approved |
                             declined and IS the workflow record: it is
                             what stops the rules re-proposing a pair
                             somebody already settled.
         :NOT_SAME_AS        a correction. Permanently blocks the pair.

  Virtuoso  owl:sameAs       the assertion, and the only one. Reached by
                             emitting AssertSameAs; withdrawn by
                             RetractSameAs.

So nothing here merges nodes. Collapsing a duplicate into its canonical
was Neo4j's substitute for reasoning it cannot do: it destroyed the
losing node, baked an identity decision irreversibly into the graph
store, and duplicated a fact Virtuoso already holds properly. Queries
needing graph traversal AND identity federate across the two stores.

A candidate asserts nothing and emits nothing. Only an approved
equivalence — a rule permitted to assert automatically, or a reviewer's
approval — becomes an AssertSameAs.

When settings.auto_merge_enabled is False, a "merge" decision becomes a
proposal instead. **Per-rule override**: a decision carrying
``force_auto_merge: True`` (stamped by the engine when the firing rule
sets ``Rule.force_auto_merge = True``) bypasses that gate. Reserved for
deterministic identifier matches — see ``rules/base.py``.
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


def entity_iri(entity_type: str, entity_id: str) -> str:
    """Stable IRI for an entity, matching the sinks' minting scheme.

    Public because the manual-review endpoint emits AssertSameAs for
    reviewer-approved merges and must mint identical IRIs.
    """
    label = _IRI_LABEL_BY_TYPE.get(entity_type, entity_type)
    return f"http://data.fontem.eu/id/{label}/{entity_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scalar(value: object) -> str | None:
    """Neo4j relationship properties hold primitives only; conflict values
    arrive as canonicalised identifier strings but are typed `object`."""
    return None if value is None else str(value)




# Six kwargs: the five-arg executor dispatch contract plus the optional
# emit collector. Bundling them into an object would hide the contract
# every action handler is written against.
#
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
        return await _execute_assert(
            driver, database, decision=decision, collect=collect,
        )

    if decision.action == "link":
        rel_type = decision.details.get("rel_type", "RELATED_TO")
        await _link(driver, database, decision=decision, rel_type=rel_type)
        return "auto_link"

    if decision.action == "flag":
        conflict = bool(decision.details.get("conflict", False))
        proposed = await _propose_candidate(
            driver, database, decision=decision, conflict=conflict
        )
        if not proposed:
            return "noop"
        return "conflict" if conflict else "flag"

    if decision.action == "enrich":
        await _enrich(driver, database, decision=decision)
        return "enrich"

    return "noop"


# Same six-kwarg dispatch contract as execute(); see the note there.
async def _execute_assert(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    collect: list[dict] | None,
) -> str:
    """A rule allowed to assert automatically: publish the equivalence.

    Nothing is written to Neo4j. An equivalence is a fact about identity,
    and identity lives in Virtuoso — emitting AssertSameAs IS the action.

    This used to collapse the pair with apoc.refactor.mergeNodes,
    destroying the losing node to simulate reasoning Neo4j cannot do. It
    also wrote a :SAME_AS edge duplicating the owl:sameAs Virtuoso
    already holds, and nothing in Neo4j ever followed that edge.

    Unlike the public execute(), this takes only what it uses: it is a
    private helper with one call site, so the uniform five-arg handler
    shape buys nothing here. `entity` and `candidate` described the pair
    only so the merge could address both nodes; the decision carries the
    ids the emit needs.
    """
    # The per-rule force_auto_merge stamp lets deterministic rules
    # (exact LEI/CIK/VAT/authority-id, GLEIF successor) assert even when
    # the global gate is OFF. See rules/base.py:Rule.force_auto_merge.
    forced = bool(decision.details.get("force_auto_merge"))
    if not settings.auto_merge_enabled and not forced:
        # Not allowed to assert on its own — this becomes a proposal, and
        # a proposal publishes nothing.
        proposed = await _propose_candidate(driver, database, decision=decision)
        return "flag" if proposed else "noop"
    await _emit_same_as_event(decision, collect)
    return "auto_assert"


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
    a_iri = entity_iri(decision.entity_type, decision.source_id)
    b_iri = entity_iri(decision.entity_type, decision.target_id)
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


async def _propose_candidate(
    driver: AsyncDriver,
    database: str,
    *,
    decision: Decision,
    conflict: bool = False,
) -> bool:  # pylint: disable=too-many-locals
    """Record a proposal on the :SAME_AS_CANDIDATE edge for review.

    This asserts nothing and emits nothing. It says a rule thinks the
    pair might be the same and here is the evidence.

    Returns False when the pair was already settled (corrected,
    asserted, or declined) so the caller records a noop rather than
    claiming a candidate was queued.

    Schema (Neo4j relationships only allow primitives + arrays of
    primitives, so the per-rule detection list is stored as three
    parallel arrays — readers zip them):
      r.detection_rules       : list[str]   ordered, no duplicates
      r.detection_confidences : list[float] aligned to detection_rules
      r.detection_dates       : list[str]   aligned (ISO timestamps)
      r.confidence            : highest of detection_confidences (summary)
      r.method                : rule_name at the max-confidence index
      r.detected_at           : detected_at at the max-confidence index
      r.status                : "pending" until a reviewer decides, then
                                "approved" or "declined". Both are
                                terminal and are what stop the rules
                                re-proposing the pair forever. On
                                approval the edge is KEPT with
                                status="approved" rather than deleted:
                                it is Neo4j's only record that the pair
                                was settled, since the assertion itself
                                lives in Virtuoso.
      r.conflict              : sticky-once-true (any rule reporting it)
      r.conflict_property     : which identifier disagreed ("lei", "vat",
                                "registered_as", ...) — set with the flag
                                and never cleared, so the reviewer can see
                                WHY a pair is contested instead of just THAT
                                it is. Was silently dropped before, leaving
                                every conflicted edge unexplained.
      r.conflict_left/right   : the two canonical values that disagreed
    """
    label = decision.entity_type
    id_key = _id_key(label)
    details = decision.details or {}
    async with driver.session(database=database) as session:
        result = await session.run(
            f"""
            MATCH (a:{label} {{{id_key}: $source_id}})
            MATCH (b:{label} {{{id_key}: $target_id}})
            // Two ways this pair is already settled and must not be
            // re-proposed, both matched in either direction because
            // neither relationship is directional in meaning.
            //   :NOT_SAME_AS  — a correction; the answer is permanent
            //   status set    — a reviewer already ruled. Without this
            //                   the deterministic rules re-propose it on
            //                   the very next sweep and the queue can
            //                   never be drained.
            //
            // There is deliberately no check for an existing :SAME_AS:
            // Neo4j holds no assertions any more. The candidate's own
            // `status` is the local record of what was decided, which is
            // what lets this run without a cross-store lookup in the hot
            // path of every rule evaluation.
            WHERE NOT EXISTS {{ (a)-[:NOT_SAME_AS]-(b) }}
              AND NOT EXISTS {{
                (a)-[d:SAME_AS_CANDIDATE]-(b)
                WHERE d.status IN ['declined', 'approved']
              }}
            MERGE (a)-[r:SAME_AS_CANDIDATE]->(b)
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
                r.status = coalesce(r.status, 'pending'),
                r.conflict = (coalesce(r.conflict, false) OR $conflict),
                // Keep the first explanation recorded; don't let a later
                // non-conflicting rule blank out why the pair is contested.
                r.conflict_property =
                  coalesce(r.conflict_property, $conflict_property),
                r.conflict_left  = coalesce(r.conflict_left,  $conflict_left),
                r.conflict_right = coalesce(r.conflict_right, $conflict_right)
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
            RETURN 1 AS proposed
            """,
            source_id=decision.source_id,
            target_id=decision.target_id,
            confidence=decision.confidence,
            rule_name=decision.rule_name,
            detected_at=_now(),
            conflict=conflict,
            conflict_property=details.get("conflicting_property"),
            conflict_left=_scalar(details.get("left")),
            conflict_right=_scalar(details.get("right")),
        )
        return await result.single() is not None


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
