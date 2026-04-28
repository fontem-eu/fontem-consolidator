"""POST /resolve — central entity-identification entry point.

Each ETL replaces its inline `MATCH (c:Company) WHERE ...` cypher
with one call to this endpoint. See `src.consolidator.resolver` for
the tiered resolution logic and the rationale.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.consolidator import resolver
from src.consolidator.neo4j.client import get_driver

router = APIRouter()


class ResolveRequest(BaseModel):
    """Attribute bag the caller knows about the entity it's looking for."""

    entity_type: Literal["Company", "Authority"]
    name: str | None = None
    country: str | None = None  # any of: ISO-3, ISO-2, full English name
    lei: str | None = None
    vat: str | None = None
    cik: str | None = None


class ResolveMatchOut(BaseModel):
    """Wire form of a single resolver match."""

    gmr_id: str
    name: str
    country: str | None
    lei: str | None
    tier: Literal["lei", "vat", "cik", "name_country", "fuzzy"]
    confidence: float


class ResolveResponse(BaseModel):
    """Wire form of the resolver result — exactly one of `match` (single
    confident hit), `candidates` (review queue), or hint=no_match."""

    hint: Literal["matched", "ambiguous", "no_match"]
    match: ResolveMatchOut | None = None
    candidates: list[ResolveMatchOut] = []
    normalised_country: str | None = None


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_endpoint(req: ResolveRequest) -> ResolveResponse:
    """Resolve an entity from a bag of attributes.

    See `src.consolidator.resolver` for the tiered logic. ETLs replace
    their inline match cypher with one POST to this endpoint."""
    if not any([req.lei, req.vat, req.cik, req.name]):
        raise HTTPException(
            status_code=400,
            detail="at least one of lei/vat/cik/name must be provided",
        )
    driver = await get_driver()
    result = await resolver.resolve(
        driver,
        settings.neo4j_database,
        entity_type=req.entity_type,
        name=req.name,
        country=req.country,
        lei=req.lei,
        vat=req.vat,
        cik=req.cik,
    )
    return ResolveResponse(
        hint=result.hint,
        match=ResolveMatchOut(**result.match.__dict__) if result.match else None,
        candidates=[ResolveMatchOut(**c.__dict__) for c in result.candidates],
        normalised_country=result.normalised_country,
    )
