"""Do we hold this fiscal number, VIES-style? (data-backlog Part 5, C3)

eForms `cbc:CompanyID` often carries a bare national id with no
schemeName -- a PT NIF, `503536717`. canon_vat only accepts the prefixed
form and returns None, the TED matcher falls back to the name and mints
a duplicate: 42,573 of 43,045 PT companies have no VAT stored, and
"Visualforma" exists three times. Before the cleaner mints anything it
can ask here under which prefix, if any, the number is already held.
Ours, so no VIES usage limits.

A (country, bare number) pair is a hypothesis. With a country there is
one; a number that already carries a prefix is its own; with neither,
one per VIES prefix whose pattern the bare number fits -- a 9-digit
number is a well-formed BE/BG/CZ/DE/EE/EL/ES/GB/LT/PT/RO VAT, and the
caller sees all of them rather than a guess. Greece is one hypothesis
with two VAT forms (EL on VIES, GR elsewhere). Both a prefixed number
and a country make two hypotheses, and the response shows both.

Per hypothesis the graph is asked what the resolver already asks, each
an index seek: Company.vat on every VAT form, Company.registered_as +
country on the bare number, Authority.national_id + country on either
spelling. One row per holder, or one unmatched row per hypothesis, so a
caller can tell "not held" from "held under DE, not PT".
"""
from __future__ import annotations

from dataclasses import dataclass

from src.consolidator import identifiers, resolver


@dataclass(frozen=True)
class Hypothesis:
    """One country the number could belong to and the VAT strings it
    would take there, VIES spelling first. `vat_forms` is empty when the
    country has a prefix but no VAT pattern (CH, NO, IS, LI) or the
    number is not VAT-shaped there (a German HRB number): the lookup
    then runs on registered_as / national_id alone."""

    country: str  # ISO-3, as the graph stores it
    number: str   # national part, punctuation stripped, upper-cased
    vat_forms: tuple[str, ...]


def normalise(raw: str | None) -> str:
    """The number as it will be looked up. Empty means "nothing to ask"."""
    return identifiers.normalise_id(raw)


def hypotheses(number: str, country: str | None = None) -> list[Hypothesis]:
    """Every (country, bare number) the input could denote.

    Empty when the number normalises to nothing, or when no country is
    given and no VIES prefix makes a well-formed VAT of it. A country
    the resolver cannot normalise is ignored here; the route rejects it
    before calling."""
    s = normalise(number)
    if not s:
        return []
    split = identifiers.split_vat(s)
    bare = split[1] if split else s
    countries: list[str] = []
    if split is not None:
        _add(countries, resolver.normalize_country(split[0]))
    _add(countries, resolver.normalize_country(country))
    if not countries:
        for prefix in identifiers.known_vat_prefixes():
            if identifiers.canon_vat(prefix + s) is not None:
                _add(countries, resolver.normalize_country(prefix))
    return [
        Hypothesis(c, bare, tuple(identifiers.vat_forms(c, bare)))
        for c in countries
    ]


def _add(countries: list[str], iso3: str | None) -> None:
    if iso3 is not None and iso3 not in countries:
        countries.append(iso3)


async def check(
    driver, database: str, *, number: str, country: str | None = None,
) -> dict:
    """The endpoint's answer: the normalised query, one row per holder
    (or per hypothesis nobody holds), and whether anything held it."""
    rows: list[dict] = []
    async with driver.session(database=database) as session:
        for h in hypotheses(number, country):
            rows.extend(await _probe(session, h))
    return {
        "query": {
            "number": normalise(number),
            "country": resolver.normalize_country(country),
        },
        "candidates": rows,
        "held": any(r["matched"] for r in rows),
    }


async def _probe(session, h: Hypothesis) -> list[dict]:
    """Ask the graph about one hypothesis. A node holding the number as
    both vat and registered_as is listed once, under the property that
    found it first (vat, then registered_as, then national_id)."""
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    primary = h.vat_forms[0] if h.vat_forms else None

    def keep(vat: str | None, entity_type: str, matched_on: str, row: dict) -> None:
        key = (entity_type, row["gmr_id"])
        if key not in seen:
            seen.add(key)
            found.append(_holder(h, vat, entity_type, matched_on, row))

    for vat in h.vat_forms:
        for row in await resolver.lookup_by_vat(session, vat):
            keep(vat, "Company", "vat", row)
    for row in await resolver.lookup_by_registered_as(session, h.number, h.country):
        keep(primary, "Company", "registered_as", row)
    spellings = [*h.vat_forms, h.number]
    for row in await resolver.lookup_authority_by_national_id(session, spellings, h.country):
        keep(primary, "Authority", "national_id", row)
    return found or [_unmatched(h, primary)]


def _holder(
    h: Hypothesis, vat: str | None, entity_type: str, matched_on: str, row: dict,
) -> dict:
    is_company = entity_type == "Company"
    return {
        "vat": vat,
        "country": h.country,
        "matched": True,
        "entity_type": entity_type,
        "gmr_id": row["gmr_id"] if is_company else None,
        "authority_id": None if is_company else row["gmr_id"],
        "name": row.get("name"),
        "matched_on": matched_on,
    }


def _unmatched(h: Hypothesis, vat: str | None) -> dict:
    return {
        "vat": vat, "country": h.country, "matched": False,
        "entity_type": None, "gmr_id": None, "authority_id": None,
        "name": None, "matched_on": None,
    }
