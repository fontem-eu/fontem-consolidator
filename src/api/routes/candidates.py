from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from src.api.lang import apply_translation, safe_lang
from src.config import settings
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
              // Each rule that has flagged this pair appends an entry to
              // r.detections; older edges (pre-multi-rule schema) won't
              // have it, so we coalesce to a single-element list built
              // from the legacy summary fields.
              coalesce(r.detections, [{{
                rule_name: r.method,
                confidence: r.confidence,
                detected_at: r.detected_at
              }}]) AS detections,
              coalesce(r.conflict, false) AS conflict,
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
        # detections list — coerce any embedded Neo4j DateTime values to ISO.
        detections = [
            {
                "rule_name": d.get("rule_name"),
                "confidence": d.get("confidence"),
                "detected_at": _iso(d.get("detected_at")),
            }
            for d in (rec["detections"] or [])
            if d.get("rule_name") is not None
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
            await session.run(
                f"""
                MATCH (a:{label} {{{id_key}: $from}})-[r:SAME_AS]-(b:{label} {{{id_key}: $to}})
                DELETE r
                """,
                **{"from": from_id, "to": to_id},
            )
            decision_type = "manual_reject"
        elif body.decision == "keep_as_related":
            await session.run(
                f"""
                MATCH (a:{label} {{{id_key}: $from}})-[r:SAME_AS]-(b:{label} {{{id_key}: $to}})
                SET r.reviewed = true, r.reviewed_at = $now, r.reviewer = $reviewer
                """,
                **{"from": from_id, "to": to_id, "now": _now(), "reviewer": body.reviewer},
            )
            decision_type = "manual_keep_related"
        else:  # merge
            await session.run(
                f"""
                MATCH (canonical:{label} {{{id_key}: $from}})
                MATCH (dup:{label} {{{id_key}: $to}})
                CALL apoc.refactor.mergeNodes([canonical, dup], {{
                  properties: "discard",
                  mergeRels: true
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

    return {"outcome": decision_type, "rule_name": rule_name}
