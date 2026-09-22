"""GET /fiscal-id/{number} — do we hold this fiscal number, VIES-style?

The hypothesis model and the graph questions live in
src.consolidator.fiscal_id; this is the wire shape. Cluster-internal
like the rest of the API. The TED cleaner (data-backlog C3) calls it
before minting a supplier from a bare `cbc:CompanyID`.
"""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.config import settings
from src.consolidator import fiscal_id
from src.consolidator.neo4j.client import get_driver
from src.consolidator.resolver import normalize_country

router = APIRouter()


class FiscalQuery(BaseModel):
    """What was asked, normalised: the number as looked up and the
    country as ISO-3 (None when none was given)."""

    number: str
    country: str | None


class FiscalCandidate(BaseModel):
    """One holder of the number under one prefix -- or, with
    matched=false, one prefix nobody holds it under.

    `vat` is the VIES spelling the hypothesis takes (the one that hit,
    for a vat match); None when the country has no VAT pattern for it.
    `country` is ISO-3. Exactly one of gmr_id / authority_id is set on a
    match, by entity_type."""

    vat: str | None
    country: str
    matched: bool
    entity_type: Literal["Company", "Authority"] | None = None
    gmr_id: str | None = None
    authority_id: str | None = None
    name: str | None = None
    matched_on: Literal["vat", "registered_as", "national_id"] | None = None


class FiscalIdResponse(BaseModel):
    query: FiscalQuery
    candidates: list[FiscalCandidate]
    #: True when any candidate matched: we hold the number somewhere.
    held: bool


@router.get(
    "/fiscal-id/{number}",
    responses={
        400: {
            "description": (
                "The number normalises to nothing (only punctuation), "
                "or `country` is not a code the resolver knows."
            ),
        },
    },
)
async def fiscal_id_lookup(
    number: str,
    country: Annotated[str | None, Query(
        description="ISO-2 or ISO-3 (EL, GR and GRC all mean Greece). "
                    "Without it every prefix the number fits is tried.",
    )] = None,
) -> FiscalIdResponse:
    """Answer whether the graph holds a fiscal number and under which prefix.

    Punctuation, spaces and case in `number` are ignored; a number that
    already carries its prefix (`PT503536717`) needs no `country`. Bare
    numbers without a country fan out to every prefix whose pattern
    fits, so `503536717` alone returns a dozen candidates -- pass the
    country when it is known.
    """
    if not fiscal_id.normalise(number):
        raise HTTPException(status_code=400, detail="number normalises to nothing")
    if country and normalize_country(country) is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown country {country!r}; use ISO-2 or ISO-3",
        )
    driver = await get_driver()
    result = await fiscal_id.check(
        driver, settings.neo4j_database, number=number, country=country,
    )
    return FiscalIdResponse(**result)
