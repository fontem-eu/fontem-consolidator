"""Internal webhook endpoint called by the Neo4j APOC trigger.

Payload shape from the trigger:
  { "label": "Company" | "Authority", "gmr_id": "...", "authority_id": "..." }

The endpoint dispatches to the engine asynchronously so the trigger call
returns immediately. Failures are logged but never propagated to the
trigger — the graph write must not be coupled to the consolidator.
"""

from fastapi import APIRouter, BackgroundTasks
from loguru import logger
from pydantic import BaseModel

from src.config import settings
from src.consolidator import engine
from src.consolidator.neo4j.client import get_driver

router = APIRouter()


class TriggerPayload(BaseModel):
    label: str
    gmr_id: str | None = None
    authority_id: str | None = None


@router.post("/webhooks/neo4j-trigger")
async def neo4j_trigger(payload: TriggerPayload, background: BackgroundTasks) -> dict:
    entity_type = payload.label
    entity_id = payload.gmr_id if entity_type == "Company" else payload.authority_id
    if not entity_id:
        logger.warning("neo4j-trigger: missing id for label={}", entity_type)
        return {"accepted": False, "reason": "missing_id"}

    async def _run() -> None:
        try:
            driver = await get_driver()
            await engine.consolidate(
                driver,
                settings.neo4j_database,
                entity_type=entity_type,
                entity_id=entity_id,
                triggered_by="apoc_trigger",
            )
        except Exception:
            logger.exception("neo4j-trigger: run failed for {}/{}", entity_type, entity_id)

    background.add_task(_run)
    return {"accepted": True, "entity_type": entity_type, "entity_id": entity_id}
