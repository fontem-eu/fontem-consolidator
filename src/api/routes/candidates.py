from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from src.api.lang import apply_translation, safe_lang
from src.config import settings
from src.consolidator import actions, eventlog
from src.consolidator.neo4j.client import get_driver

router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: Any) -> str | None:
    """Coerce Neo4j DateTime / Python datetime / string into a JSON-friendly ISO string."""
    if value is None:
        return None
    if hasattr(value, "iso_format"):  # neo4j.time.DateTime
        return value.iso_format()
    if hasattr(value, "isoformat"):  # datetime.datetime
        return value.isoformat()
    return str(value)


@router.get("/candidates")
async def list_candidates(
    response: Response,
    entity_type: str | None = Query(default=None),
    status: str = Query(default="pending", pattern="^(pending|declined)$"),
    limit: int = Query(default=50, le=500),
    cursor: str | None = Query(default=None),
    lang: str | None = Query(default=None),
):
    """List :SAME_AS_CANDIDATE proposals awaiting a decision.

    Candidates are proposals, not assertions — nothing here has been
    published to Virtuoso. Approving one writes the :SAME_AS edge and
    emits AssertSameAs; that is the only path from this queue into the
    graph as an equivalence.

    `status` defaults to "pending". "declined" lists what reviewers have
    already turned down, which is kept rather than deleted because it is
    what stops the rules re-proposing those pairs.

    `lang` (ISO-639-1) swaps the `name` field of each entity for its
    translated counterpart (`name_<lang>`) when one is present. Missing
    translations or non-Authority entities fall through to the stored
    `name` unchanged.
    """
    effective_lang = safe_lang(lang)
    driver = await get_driver()
    where = ["coalesce(r.status, 'pending') = $status"]
    params: dict = {"status": status, "limit": limit}
    if entity_type:
        where.append("(from:" + entity_type + " AND to:" + entity_type + ")")
    if cursor:
        where.append("r.detected_at < $cursor")
        params["cursor"] = cursor

    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            f"""
            MATCH (a)-[r:SAME_AS_CANDIDATE]->(b)
            WHERE {" AND ".join(where)}
            RETURN
              r.confidence AS confidence,
              r.method AS rule_name,
              r.detected_at AS detected_at,
              // Per-rule detections are stored as three parallel arrays
              // because Neo4j relationship props can't hold list<map>.
              // Coalesce against legacy edges (single rule, summary only)
              // by falling back to a one-element list built from r.method.
              coalesce(r.detection_rules,       [r.method])      AS det_rules,
              coalesce(r.detection_confidences, [r.confidence])  AS det_confs,
              coalesce(r.detection_dates,       [r.detected_at]) AS det_dates,
              coalesce(r.conflict, false) AS conflict,
              r.conflict_property AS conflict_property,
              r.conflict_left AS conflict_left,
              r.conflict_right AS conflict_right,
              labels(a) AS a_labels, a {{.*}} AS a_props,
              labels(b) AS b_labels, b {{.*}} AS b_props
            ORDER BY r.detected_at DESC
            LIMIT $limit
            """,
            **params,
        )
        rows = [record async for record in result]
    out = []
    for rec in rows:
        a_type = "Company" if "Company" in rec["a_labels"] else "Authority"
        source_id = rec["a_props"].get("gmr_id") or rec["a_props"].get("authority_id")
        target_id = rec["b_props"].get("gmr_id") or rec["b_props"].get("authority_id")
        # Zip the three parallel arrays back into a list of objects for
        # the wire format. Skip entries with a null rule_name (defensive
        # against half-populated legacy edges).
        detections = [
            {"rule_name": rn, "confidence": rc, "detected_at": _iso(dt)}
            for rn, rc, dt in zip(
                rec["det_rules"] or [],
                rec["det_confs"] or [],
                rec["det_dates"] or [],
            )
            if rn is not None
        ]
        out.append(
            {
                "from_id": source_id,
                "to_id": target_id,
                "entity_type": a_type,
                # Legacy summary fields — still populated, kept for
                # backward compat. UI can prefer `detections` when
                # rendering multi-rule evidence.
                "rule_name": rec["rule_name"],
                "confidence": rec["confidence"],
                "detected_at": _iso(rec["detected_at"]),
                "detections": detections,
                "conflict": rec["conflict"],
                # Why the pair is contested, not merely that it is: which
                # identifier disagreed and the two canonical values.
                "conflict_property": rec["conflict_property"],
                "conflict_left": rec["conflict_left"],
                "conflict_right": rec["conflict_right"],
                "source_entity": {k: _iso(v) if hasattr(v, "iso_format") else v
                                  for k, v in apply_translation(rec["a_props"], effective_lang).items()},
                "target_entity": {k: _iso(v) if hasattr(v, "iso_format") else v
                                  for k, v in apply_translation(rec["b_props"], effective_lang).items()},
            }
        )
    if len(out) == limit and out[-1]["detected_at"]:
        response.headers["X-Next-Cursor"] = out[-1]["detected_at"]
    return out


class DecideBody(BaseModel):
    """A reviewer's verdict on a proposal.

    "approve" is the only value that asserts anything. The legacy names
    are still accepted so the deploy does not have to be simultaneous
    with the web release: "merge" -> approve, "reject" -> decline.
    """

    decision: Literal[
        "approve", "decline", "keep_as_related", "merge", "reject",
    ]
    reviewer: str
    note: str | None = None

    @property
    def verdict(self) -> str:
        return {"merge": "approve", "reject": "decline"}.get(
            self.decision, self.decision,
        )


@router.post("/candidates/{from_id}/{to_id}/decide")
async def decide(from_id: str, to_id: str, body: DecideBody):
    driver = await get_driver()
    event_seq: int | None = None
    async with driver.session(database=settings.neo4j_database) as session:
        # Find the candidate between these two nodes regardless of direction
        result = await session.run(
            """
            MATCH (a)-[r:SAME_AS_CANDIDATE]-(b)
            WHERE (a.gmr_id = $from OR a.authority_id = $from)
              AND (b.gmr_id = $to OR b.authority_id = $to)
            RETURN a, b, r, labels(a)[0] AS label
            LIMIT 1
            """,
            **{"from": from_id, "to": to_id},
        )
        rec = await result.single()
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail="no SAME_AS_CANDIDATE found between these entities",
            )
        if rec["r"].get("status") == "declined":
            raise HTTPException(
                status_code=409,
                detail="candidate already declined; use the correction "
                       "endpoint if an assertion needs undoing",
            )
        label = rec["label"]
        id_key = "gmr_id" if label == "Company" else "authority_id"
        rule_name = rec["r"].get("method", "unknown")
        confidence = rec["r"].get("confidence", 0.0)

        verdict = body.verdict
        common = {
            "from": from_id, "to": to_id, "now": _now(),
            "reviewer": body.reviewer, "note": body.note,
        }

        if verdict == "decline":
            # The candidate is KEPT, marked terminal. Deleting it would
            # be worse than useless: the rules are deterministic and the
            # sweeper re-runs them over every entity forever, so the very
            # next pass would re-propose the identical pair. The declined
            # edge IS the memory of the decision.
            #
            # This is not a :NOT_SAME_AS. Nothing was ever asserted here,
            # so there is nothing to retract, and no event is emitted.
            await session.run(
                f"""
                MATCH (a:{label} {{{id_key}: $from}})-[r:SAME_AS_CANDIDATE]-(b:{label} {{{id_key}: $to}})
                SET r.status = 'declined', r.decided_at = $now,
                    r.reviewer = $reviewer, r.note = $note
                """,
                **common,
            )
            decision_type = "manual_reject"

        elif verdict == "keep_as_related":
            # "Related but not the same" — a real edge, and a decline of
            # the equivalence. Both, together.
            await session.run(
                f"""
                MATCH (a:{label} {{{id_key}: $from}})-[r:SAME_AS_CANDIDATE]-(b:{label} {{{id_key}: $to}})
                SET r.status = 'declined', r.decided_at = $now,
                    r.reviewer = $reviewer, r.note = $note
                MERGE (a)-[rel:RELATED_TO]->(b)
                SET rel.reviewed = true, rel.reviewed_at = $now,
                    rel.reviewer = $reviewer, rel.source = 'manual_review'
                """,
                **common,
            )
            decision_type = "manual_keep_related"

        else:  # approve
            # The approval IS the assertion. Both nodes survive — an
            # assertion has to stay correctable, and :NOT_SAME_AS can
            # only undo something that still exists.
            asserted = await actions.assert_same_as(
                driver, settings.neo4j_database,
                label=label, source_id=from_id, target_id=to_id,
                method=rule_name, confidence=float(confidence or 1.0),
                origin="review", reviewer=body.reviewer,
            )
            if not asserted:
                raise HTTPException(
                    status_code=409,
                    detail="a :NOT_SAME_AS correction blocks this pair",
                )
            await session.run(
                f"""
                MATCH (a:{label} {{{id_key}: $from}})-[r:SAME_AS_CANDIDATE]-(b:{label} {{{id_key}: $to}})
                DELETE r
                """,
                **{"from": from_id, "to": to_id},
            )
            decision_type = "manual_merge"

            # Publishing the approved equivalence. emit_assert_same_as
            # absorbs its own failures and returns None; surface that in
            # the response rather than reporting a silent success, because
            # an approval that never reached Virtuoso is invisible.
            event_seq = await eventlog.emit_assert_same_as(
                a_iri=actions.entity_iri(label, from_id),
                b_iri=actions.entity_iri(label, to_id),
                confidence=float(confidence or 1.0),
                method=rule_name,
                rule=rule_name,
                domain=label.lower(),
            )

        # Log the manual decision
        await session.run(
            """
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
              review_note: $note
            })
            """,
            decision_id=str(uuid4()),
            decided_at=_now(),
            decision_type=decision_type,
            rule_name=rule_name,
            confidence=confidence,
            source_id=from_id,
            target_id=to_id,
            entity_type=label,
            reviewer=body.reviewer,
            note=body.note,
        )

    return {
        "outcome": decision_type,
        "rule_name": rule_name,
        # None on a merge means the owl:sameAs assertion did NOT reach the
        # event log; the graph merge still happened. Operators need to see it.
        "event_seq": event_seq,
        # None on an approval means the owl:sameAs assertion did NOT reach
        # the event log; the :SAME_AS edge still exists. Operators need it.
        "projected": decision_type != "manual_merge" or event_seq is not None,
    }


class CorrectBody(BaseModel):
    """Undo an equivalence the platform actually published."""

    reason: str
    reviewer: str


@router.post("/same-as/{from_id}/{to_id}/correct")
async def correct(from_id: str, to_id: str, body: CorrectBody):
    """Record a :NOT_SAME_AS correction against an asserted :SAME_AS.

    This is the mistake path, and it is deliberately separate from
    declining a candidate. Declining rejects a proposal that was never
    published; correcting withdraws a claim that was. Only the second
    needs a retraction event, because only the second put a triple in
    Virtuoso that has to come back out.

    The correction is permanent and outranks every rule: the
    :NOT_SAME_AS edge blocks the pair from being re-asserted or even
    re-proposed, which matters because the rule that got it wrong is
    deterministic and would otherwise reach the same conclusion on the
    next sweep.
    """
    driver = await get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """
            MATCH (a)-[r:SAME_AS]-(b)
            WHERE (a.gmr_id = $from OR a.authority_id = $from)
              AND (b.gmr_id = $to OR b.authority_id = $to)
            RETURN a, b, r, labels(a)[0] AS label
            LIMIT 1
            """,
            **{"from": from_id, "to": to_id},
        )
        rec = await result.single()
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail="no asserted :SAME_AS between these entities; a "
                       "proposal that was never approved is declined via "
                       "/candidates/{from}/{to}/decide, not corrected",
            )
        label = rec["label"]
        id_key = "gmr_id" if label == "Company" else "authority_id"
        retracted_method = rec["r"].get("method", "unknown")

        # Drop the assertion, record the correction, and clear any
        # candidate for the pair so the queue does not re-offer it.
        await session.run(
            f"""
            MATCH (a:{label} {{{id_key}: $from}})
            MATCH (b:{label} {{{id_key}: $to}})
            OPTIONAL MATCH (a)-[s:SAME_AS]-(b)
            DELETE s
            WITH a, b
            OPTIONAL MATCH (a)-[c:SAME_AS_CANDIDATE]-(b)
            DELETE c
            WITH a, b
            MERGE (a)-[n:NOT_SAME_AS]->(b)
            SET n.decided_at = $now, n.reviewer = $reviewer,
                n.reason = $reason, n.retracted_method = $method
            """,
            **{
                "from": from_id, "to": to_id, "now": _now(),
                "reviewer": body.reviewer, "reason": body.reason,
                "method": retracted_method,
            },
        )

        await session.run(
            """
            CREATE (dl:DecisionLog {
              decision_id: $decision_id,
              decided_at: $decided_at,
              decision_type: 'manual_correction',
              rule_name: $rule_name,
              source_id: $source_id,
              target_id: $target_id,
              entity_type: $entity_type,
              reviewer: $reviewer,
              review_note: $note
            })
            """,
            decision_id=str(uuid4()),
            decided_at=_now(),
            rule_name=retracted_method,
            source_id=from_id,
            target_id=to_id,
            entity_type=label,
            reviewer=body.reviewer,
            note=body.reason,
        )

    # Withdraw the published triple. A failure here leaves a wrong
    # owl:sameAs standing in Virtuoso even though Neo4j is corrected,
    # so the caller is told rather than reassured.
    event_seq = await eventlog.emit_retract_same_as(
        a_iri=actions.entity_iri(label, from_id),
        b_iri=actions.entity_iri(label, to_id),
        reason=body.reason,
        reviewer=body.reviewer,
        retracted_method=retracted_method,
        domain=label.lower(),
    )
    return {
        "outcome": "manual_correction",
        "retracted_method": retracted_method,
        "event_seq": event_seq,
        "retracted": event_seq is not None,
    }
