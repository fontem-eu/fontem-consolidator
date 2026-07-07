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
    # National business-register ID (GLEIF RegistrationAuthorityEntityID).
    # Only resolved together with country. Forward-prep.
    registered_as: str | None = None


class ResolveMatchOut(BaseModel):
    """Wire form of a single resolver match."""

    gmr_id: str
    name: str
    country: str | None
    lei: str | None
    tier: Literal[
        "lei", "vat", "cik", "registered_as", "name_country", "fuzzy",
    ]
    confidence: float


class ResolveResponse(BaseModel):
    """Wire form of the resolver result — exactly one of `match` (single
    confident hit), `candidates` (review queue), or hint=no_match."""

    hint: Literal["matched", "ambiguous", "no_match"]
    match: ResolveMatchOut | None = None
    candidates: list[ResolveMatchOut] = []
    normalised_country: str | None = None


def _to_response(result) -> "ResolveResponse":
    return ResolveResponse(
        hint=result.hint,
        match=ResolveMatchOut(**result.match.__dict__) if result.match else None,
        candidates=[ResolveMatchOut(**c.__dict__) for c in result.candidates],
        normalised_country=result.normalised_country,
    )


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_endpoint(req: ResolveRequest) -> ResolveResponse:
    """Resolve an entity from a bag of attributes.

    See `src.consolidator.resolver` for the tiered logic. ETLs replace
    their inline match cypher with one POST to this endpoint."""
    if not any([req.lei, req.vat, req.cik, req.registered_as, req.name]):
        raise HTTPException(
            status_code=400,
            detail=("at least one of lei/vat/cik/registered_as/name "
                    "must be provided"),
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
        registered_as=req.registered_as,
    )
    return _to_response(result)


# Cap on per-batch size. The resolver's per-tier work is mostly
# index-scoped lookups, but unbounded batches risk holding an HTTP
# connection open for minutes — the 200 limit lines up with what each
# ETL chunks at internally (BATCH_SIZE=500 falls within reasonable
# range for two requests).
_BATCH_LIMIT = 200


class BatchResolveRequest(BaseModel):
    """Many resolutions in one HTTP round-trip.

    Each request item carries the same shape as a single /resolve
    body. Useful for ETLs that touch tens of thousands of entities
    where the per-row HTTP overhead dominates wall-time."""

    entity_type: Literal["Company", "Authority"]
    rows: list["BatchResolveRow"]


class BatchResolveRow(BaseModel):
    """One row of a batch resolve. Per-row entity_type is not
    accepted — the batch is homogeneous by design."""

    name: str | None = None
    country: str | None = None
    lei: str | None = None
    vat: str | None = None
    cik: str | None = None
    registered_as: str | None = None


class BatchResolveResponse(BaseModel):
    """Per-row results, in the same order as the input."""

    results: list[ResolveResponse]


# Forward-ref resolution for Pydantic
BatchResolveRequest.model_rebuild()


@router.post("/resolve/batch", response_model=BatchResolveResponse)
async def resolve_batch(req: BatchResolveRequest) -> BatchResolveResponse:
    """Resolve N attribute bags against the graph in one round-trip.

    Returns one ResolveResponse per input row, in input order. Empty
    rows (no attributes at all) are surfaced as `no_match` so callers
    can rely on positional alignment."""
    if not req.rows:
        return BatchResolveResponse(results=[])
    if len(req.rows) > _BATCH_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"batch size {len(req.rows)} exceeds limit {_BATCH_LIMIT}",
        )
    driver = await get_driver()
    out: list[ResolveResponse] = []
    for row in req.rows:
        if not any([row.lei, row.vat, row.cik,
                    row.registered_as, row.name]):
            out.append(ResolveResponse(hint="no_match"))
            continue
        result = await resolver.resolve(
            driver,
            settings.neo4j_database,
            entity_type=req.entity_type,
            name=row.name,
            country=row.country,
            lei=row.lei,
            vat=row.vat,
            cik=row.cik,
            registered_as=row.registered_as,
        )
        out.append(_to_response(result))
    return BatchResolveResponse(results=out)
