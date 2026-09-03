from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from src.api.lang import apply_translation, safe_lang
from src.config import settings
from src.consolidator import eventlog
from src.consolidator.actions import entity_iri
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
    reviewed: bool = Query(default=False),
    limit: int = Query(default=50, le=500),
    cursor: str | None = Query(default=None),
    lang: str | None = Query(default=None),
):
    """List SAME_AS review candidates.

    `lang` (ISO-639-1) swaps the `name` field of each entity for its
    translated counterpart (`name_<lang>`) when one is present. Missing
    translations or non-Authority entities fall through to the stored
    `name` unchanged.
    """
    effective_lang = safe_lang(lang)
    driver = await get_driver()
    where = ["r.reviewed = $reviewed"]
    params: dict = {"reviewed": reviewed, "limit": limit}
    if entity_type:
        where.append("(from:" + entity_type + " AND to:" + entity_type + ")")
    if cursor:
        where.append("r.detected_at < $cursor")
        params["cursor"] = cursor

    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            f"""
            MATCH (a)-[r:SAME_AS]->(b)
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
    decision: Literal["merge", "reject", "keep_as_related"]
    reviewer: str
    note: str | None = None


@router.post("/candidates/{from_id}/{to_id}/decide")
async def decide(from_id: str, to_id: str, body: DecideBody):
    driver = await get_driver()
    event_seq: int | None = None
    async with driver.session(database=settings.neo4j_database) as session:
        # Find the SAME_AS between these two nodes regardless of direction
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
            raise HTTPException(status_code=404, detail="no SAME_AS edge found between these entities")
        label = rec["label"]
        id_key = "gmr_id" if label == "Company" else "authority_id"
        rule_name = rec["r"].get("method", "unknown")
        confidence = rec["r"].get("confidence", 0.0)

        if body.decision == "reject":
            # Deleting the :SAME_AS is not enough — the next sweep re-runs the
            # same rules and MERGEs it straight back, so a rejected pair would
            # reappear in the queue forever. The :NOT_SAME_AS edge is the
            # durable record of the reviewer's veto; actions._flag_same_as and
            # actions._merge both refuse to act on a pair that carries one.
            await session.run(
                f"""
                MATCH (a:{label} {{{id_key}: $from}})
                MATCH (b:{label} {{{id_key}: $to}})
                OPTIONAL MATCH (a)-[r:SAME_AS]-(b)
                DELETE r
                MERGE (a)-[n:NOT_SAME_AS]->(b)
                SET n.decided_at = $now,
                    n.reviewer = $reviewer,
                    n.note = $note,
                    n.decision = 'reject',
                    n.method_at_review = $rule_name,
                    n.confidence_at_review = $confidence
                """,
                **{
                    "from": from_id, "to": to_id, "now": _now(),
                    "reviewer": body.reviewer, "note": body.note,
                    "rule_name": rule_name, "confidence": confidence,
                },
            )
            decision_type = "manual_reject"
        elif body.decision == "keep_as_related":
            # "Keep as related" means NOT the same entity but connected. It used
            # to set reviewed=true on the :SAME_AS edge and leave it there,
            # which was actively dangerous: gds/wcc_collapse projects exactly
            # `reviewed=true AND conflict=false` SAME_AS edges and merges the
            # component with force_auto_merge — so answering "these are merely
            # related" would have deleted one of the two nodes. Move the pair
            # onto :RELATED_TO and record the same-entity veto.
            await session.run(
                f"""
                MATCH (a:{label} {{{id_key}: $from}})
                MATCH (b:{label} {{{id_key}: $to}})
                OPTIONAL MATCH (a)-[r:SAME_AS]-(b)
                DELETE r
                MERGE (a)-[rel:RELATED_TO]->(b)
                SET rel.reviewed = true, rel.reviewed_at = $now,
                    rel.reviewer = $reviewer, rel.source = 'manual_review'
                MERGE (a)-[n:NOT_SAME_AS]->(b)
                SET n.decided_at = $now,
                    n.reviewer = $reviewer,
                    n.note = $note,
                    n.decision = 'keep_as_related',
                    n.method_at_review = $rule_name,
                    n.confidence_at_review = $confidence
                """,
                **{
                    "from": from_id, "to": to_id, "now": _now(),
                    "reviewer": body.reviewer, "note": body.note,
                    "rule_name": rule_name, "confidence": confidence,
                },
            )
            decision_type = "manual_keep_related"
        else:  # merge
            await session.run(
                f"""
                MATCH (canonical:{label} {{{id_key}: $from}})
                MATCH (dup:{label} {{{id_key}: $to}})
                // Same reason as actions._merge: without produceSelfRel the
                // SAME_AS edge between the pair survives as a self-loop on the
                // merged node. The manual-review path merges the same way the
                // automatic one does, so it needs the same flag.
                CALL apoc.refactor.mergeNodes([canonical, dup], {{
                  properties: "discard",
                  mergeRels: true,
                  produceSelfRel: false
                }}) YIELD node
                WITH node
                CREATE (e:MergeEvent {{
                  canonical_id: $from, merged_id: $to, merged_at: $now,
                  method: $rule_name, entity_type: $label, reviewer: $reviewer
                }})
                RETURN node
                """,
                **{
                    "from": from_id,
                    "to": to_id,
                    "now": _now(),
                    "rule_name": rule_name,
                    "label": label,
                    "reviewer": body.reviewer,
                },
            )
            decision_type = "manual_merge"

            # The reviewer approved the equivalence — this is the second of the
            # two routes allowed to assert owl:sameAs (see actions.py's
            # emission contract). The duplicate node is gone after the merge
            # above, so without this event its IRI would dangle in Virtuoso.
            # emit_assert_same_as absorbs its own failures and returns None;
            # surface that in the response rather than silently reporting
            # success, because a lost approval is invisible otherwise.
            event_seq = await eventlog.emit_assert_same_as(
                a_iri=entity_iri(label, from_id),
                b_iri=entity_iri(label, to_id),
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
        "projected": decision_type != "manual_merge" or event_seq is not None,
    }
