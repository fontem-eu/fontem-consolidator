from fastapi import APIRouter, HTTPException

from src.config import settings
from src.consolidator import engine
from src.consolidator.neo4j.client import get_driver

router = APIRouter()


@router.post("/consolidate/company/{gmr_id}")
async def consolidate_company(gmr_id: str):
    driver = await get_driver()
    result = await engine.consolidate(
        driver,
        settings.neo4j_database,
        entity_type="Company",
        entity_id=gmr_id,
        triggered_by="api",
    )
    if result.run_id is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="run failed")
    return {
        "run_id": result.run_id,
        "entity_type": result.entity_type,
        "entity_id": result.entity_id,
        "decisions": result.decisions,
        "rules_fired": result.rules_fired,
    }


@router.post("/consolidate/authority/{authority_id}")
async def consolidate_authority(authority_id: str):
    driver = await get_driver()
    result = await engine.consolidate(
        driver,
        settings.neo4j_database,
        entity_type="Authority",
        entity_id=authority_id,
        triggered_by="api",
    )
    return {
        "run_id": result.run_id,
        "entity_type": result.entity_type,
        "entity_id": result.entity_id,
        "decisions": result.decisions,
        "rules_fired": result.rules_fired,
    }
