"""Manual review queue for non-SAME_AS relationships.

The /candidates endpoint reviews `:SAME_AS` edges (entity dedup).
This endpoint reviews edges that the new resolver-driven ETLs
write with `reviewed=false`:

  - `:REPRESENTS`  — Lobbyist → Company (lobbying register)
  - `:SANCTIONED`  — Company → SanctionedEntity (EU consolidated list)

Different action vocabulary from /candidates: a SAME_AS review picks
between {merge, reject, keep_as_related}; a relationship review just
confirms (`accept`) or rejects (`reject`) the relationship as
written. There's no "merge" because we're not asserting two nodes
are the same entity — only that one stands in some relation to the
other.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.config import settings
from src.consolidator.neo4j.client import get_driver

router = APIRouter()


SUPPORTED_TYPES = {"REPRESENTS", "SANCTIONED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "iso_format"):  # neo4j.time.DateTime
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


@router.get("/relationships")
async def list_relationships(
    rel_type: Literal["REPRESENTS", "SANCTIONED"] = Query(...),
    reviewed: bool = Query(default=False),
    limit: int = Query(default=50, le=500),
):
    """List relationship-review candidates for one edge type.

    Each row carries the source/target entity properties plus the
    edge metadata (tier, confidence, method, detected_at) so the
    UI can render a self-contained card without extra lookups."""
    if rel_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported rel_type {rel_type}")

    driver = await get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            f"""
            MATCH (a)-[r:{rel_type}]->(b)
            WHERE r.reviewed = $reviewed
            RETURN
              labels(a) AS a_labels, a {{.*}} AS a_props,
              labels(b) AS b_labels, b {{.*}} AS b_props,
              r.confidence AS confidence,
              r.tier       AS tier,
              r.method     AS method,
              r.detected_at AS detected_at,
              elementId(r) AS edge_id
            ORDER BY coalesce(r.detected_at, datetime()) DESC
            LIMIT $limit
            """,
            reviewed=reviewed, limit=limit,
        )
        rows = [record async for record in result]

    out = []
    for rec in rows:
        out.append({
            "rel_type": rel_type,
            "edge_id": rec["edge_id"],
            "source": {
                "labels": rec["a_labels"],
                "id": _entity_id(rec["a_props"]),
                "name": rec["a_props"].get("name"),
                "country": rec["a_props"].get("country") or rec["a_props"].get("country_iso"),
                "props": _coerce_props(rec["a_props"]),
            },
            "target": {
                "labels": rec["b_labels"],
                "id": _entity_id(rec["b_props"]),
                "name": rec["b_props"].get("name"),
                "country": rec["b_props"].get("country") or rec["b_props"].get("nationality"),
                "props": _coerce_props(rec["b_props"]),
            },
            "tier": rec["tier"],
            "confidence": rec["confidence"],
            "method": rec["method"],
            "detected_at": _iso(rec["detected_at"]),
        })
    return out


def _entity_id(props: dict) -> str | None:
    """Return the canonical id field for whatever entity type props came from."""
    return (
        props.get("gmr_id")
        or props.get("authority_id")
        or props.get("entity_id")        # SanctionedEntity
        or props.get("tr_id")            # Lobbyist
    )


def _coerce_props(props: dict) -> dict:
    """JSON-friendly copy of node properties (datetimes → ISO strings)."""
    return {k: _iso(v) if hasattr(v, "iso_format") else v for k, v in props.items()}


class DecideRelationshipBody(BaseModel):
    """Reviewer decision on a relationship-review candidate.

    `accept` marks the edge `reviewed=true` and records the reviewer
    on the edge; the relationship stands. `reject` deletes the edge.
    Either way a :DecisionLog entry is created so the audit trail
    survives the edge being removed."""

    decision: Literal["accept", "reject"]
    reviewer: str
    note: str | None = None


@router.post("/relationships/{edge_id}/decide")
async def decide_relationship(edge_id: str, body: DecideRelationshipBody):
    """Accept or reject a single relationship review candidate."""
    driver = await get_driver()
    now = _now()
    async with driver.session(database=settings.neo4j_database) as session:
        # Fetch the edge so we know what we're acting on (the rel type
        # is implicit in the edge_id but we still need source/target
        # props for the audit log).
        result = await session.run(
            """
            MATCH (a)-[r]->(b) WHERE elementId(r) = $edge_id
            RETURN type(r) AS rel_type,
                   labels(a)[0] AS a_label, a {.*} AS a_props,
                   labels(b)[0] AS b_label, b {.*} AS b_props,
                   r.confidence AS confidence, r.tier AS tier,
                   r.method AS method
            """,
            edge_id=edge_id,
        )
        rec = await result.single()
        if rec is None:
            raise HTTPException(status_code=404, detail="edge not found")
        if rec["rel_type"] not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"edge type {rec['rel_type']} is not reviewable here — "
                       "use /candidates for SAME_AS",
            )

        source_id = _entity_id(rec["a_props"])
        target_id = _entity_id(rec["b_props"])

        if body.decision == "reject":
            await session.run(
                "MATCH ()-[r]->() WHERE elementId(r) = $edge_id DELETE r",
                edge_id=edge_id,
            )
            decision_type = "manual_reject_relationship"
        else:  # accept
            await session.run(
                """
                MATCH ()-[r]->() WHERE elementId(r) = $edge_id
                SET r.reviewed   = true,
                    r.reviewed_at = $now,
                    r.reviewer    = $reviewer
                """,
                edge_id=edge_id, now=now, reviewer=body.reviewer,
            )
            decision_type = "manual_accept_relationship"

        # Audit trail — survives the edge being deleted on reject.
        await session.run(
            """
            CREATE (dl:DecisionLog {
              decision_id: $decision_id,
              decided_at: $decided_at,
              decision_type: $decision_type,
              rule_name: coalesce($method, $rel_type),
              confidence: coalesce($confidence, 0.0),
              source_id: $source_id,
              target_id: $target_id,
              entity_type: $rel_type,
              reviewer: $reviewer,
              review_note: $note
            })
            """,
            decision_id=str(uuid4()),
            decided_at=now,
            decision_type=decision_type,
            method=rec["method"],
            confidence=rec["confidence"],
            source_id=source_id,
            target_id=target_id,
            rel_type=rec["rel_type"],
            reviewer=body.reviewer,
            note=body.note,
        )

    return {
        "outcome": decision_type,
        "rel_type": rec["rel_type"],
        "edge_id": edge_id,
    }
