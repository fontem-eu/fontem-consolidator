"""POST /events/dispatch — webhook from consolidator-trigger.

Receives one event from ``events.entity_events`` at a time and
maps it to a ``engine.consolidate(entity_type, entity_id)`` call.

Idempotent: redelivering the same event re-runs consolidation,
which is itself MERGE-based and safe to repeat.

The trigger only POSTs events whose ``event_type`` is in its
``INPUT_TYPES`` set (Upsert*/Delete*). The consolidator's own
outputs (AssertSameAs, RetractSameAs) and control events
(BeginGraphReplace, EndGraphReplace) never reach this endpoint —
the cycle is broken at the trigger.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from src.config import settings
from src.consolidator import engine
from src.consolidator.neo4j.client import get_driver

router = APIRouter()

# Maps event_type → (entity_type, payload_id_field).
#
# Filing events trigger consolidation of the *parent* Company —
# new financials may shift the dedup signal even if the company
# row itself didn't change.
#
# Sanctions and control events are intentionally absent: the
# consolidator has no rules for SanctionedEntity today, and
# Begin/End graph-replace markers carry no entity to consolidate.
# The trigger is expected to filter those out before posting; if
# one slips through we 200-noop instead of erroring.
EVENT_DISPATCH: dict[str, tuple[str, str] | None] = {
    "UpsertCompany":   ("Company",   "gmr_id"),
    "DeleteCompany":   None,
    "UpsertContract":  ("Contract",  "ted_notice_id"),
    "DeleteContract":  None,
    "UpsertAuthority": ("Authority", "authority_id"),
    "DeleteAuthority": None,
    "UpsertFiling":    ("Company",   "gmr_id"),
    "DeleteFiling":    None,
}


class DispatchRequest(BaseModel):
    seq: int
    event_type: str
    iri: str
    domain: str
    payload: dict[str, Any]
    batch_id: str | None = None
    producer: str | None = None


@router.post("/events/dispatch")
async def dispatch(req: DispatchRequest):
    mapping = EVENT_DISPATCH.get(req.event_type)
    if mapping is None:
        return {
            "outcome": "noop",
            "reason": f"no consolidation mapping for {req.event_type}",
        }

    entity_type, id_field = mapping
    entity_id = req.payload.get(id_field)
    if not entity_id:
        raise HTTPException(
            status_code=400,
            detail=f"payload missing required key {id_field!r} "
                   f"for event_type {req.event_type}",
        )

    driver = await get_driver()
    result = await engine.consolidate(
        driver,
        settings.neo4j_database,
        entity_type=entity_type,
        entity_id=str(entity_id),
        triggered_by=f"event:{req.seq}",
    )
    logger.info(
        "dispatch seq={seq} type={t} → {entity_type}/{entity_id} "
        "({rules} rules, {decisions} decisions)",
        seq=req.seq, t=req.event_type,
        entity_type=result.entity_type, entity_id=result.entity_id,
        rules=result.rules_fired, decisions=len(result.decisions),
    )
    return {
        "outcome": "consolidated",
        "run_id": result.run_id,
        "entity_type": result.entity_type,
        "entity_id": result.entity_id,
        "rules_fired": result.rules_fired,
        "decisions": len(result.decisions),
    }
