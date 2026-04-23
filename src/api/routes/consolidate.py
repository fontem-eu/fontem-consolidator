from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.consolidator import engine
from src.consolidator.neo4j.client import get_driver

router = APIRouter()


class BatchRequest(BaseModel):
    entity_type: str  # "Company" | "Authority" | "Contract"
    ids: list[str]
    triggered_by: str = "batch"
    exclude_rule_prefix: str | None = None  # e.g. "gds_" for fast bulk scans
    # Per-request override for the gmr-linguistics translation backend
    # (e.g. "mistral", "nllb-local"). `None` → the consolidator pod's
    # configured default. Useful for side-by-side quality/speed comparisons
    # without redeploying the service.
    translation_backend: str | None = None


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


@router.post("/consolidate/batch")
async def consolidate_batch(req: BatchRequest):
    """Run consolidation against many entities in one call. Used by ETL hooks."""
    if req.entity_type not in ("Company", "Authority", "Contract"):
        raise HTTPException(status_code=400, detail=f"unknown entity_type {req.entity_type}")
    driver = await get_driver()
    summary = {"processed": 0, "merged": 0, "linked": 0, "flagged": 0, "conflicts": 0}
    run_ids: list[str] = []
    for entity_id in req.ids:
        result = await engine.consolidate(
            driver,
            settings.neo4j_database,
            entity_type=req.entity_type,
            entity_id=entity_id,
            triggered_by=req.triggered_by,
            exclude_rule_prefix=req.exclude_rule_prefix,
            translation_backend=req.translation_backend,
        )
        run_ids.append(result.run_id)
        summary["processed"] += 1
        for d in result.decisions:
            outcome = d.get("outcome", "")
            if outcome == "auto_merge":
                summary["merged"] += 1
            elif outcome == "auto_link":
                summary["linked"] += 1
            elif outcome == "conflict":
                summary["conflicts"] += 1
            elif outcome == "flag":
                summary["flagged"] += 1
    return {"run_ids": run_ids, **summary}
