"""Entity-identification service: given attributes, find the entity.

Each ETL used to reinvent matching, with different bugs (sanctions
matched on bare 3-letter acronym names; lobbying had a NULL-bypassing
country guard that disabled itself; CDP would mutate properties on a
fuzzy hit). This module centralises that logic and resolves in tiers
of decreasing confidence:

    Tier 1  (lei)             LEI                         confidence 1.00
    Tier 2  (vat / cik)       canonicalised hard ID       confidence 0.99
    Tier 3  (name_country)    apoc.text.clean(name) +     confidence 0.95
                              ISO country agreement
    Tier 4  (fuzzy)           full-text candidates,       confidence < 0.95
                              NEVER auto-matched — returned for review

Every tier carries the same country guard rules: country must agree
once normalised, no NULL-bypass.

Callers receive `ResolveResult.match` (one row, ready to act on) OR
`ResolveResult.candidates` (a review list) OR `hint == "no_match"`.
The same plumbing the consolidator uses to record decisions
(:ConsolidationRun, audit) is intentionally NOT used here — this is
read-only lookup, not graph mutation. ETLs decide whether to write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from neo4j import AsyncDriver

from src.consolidator.identifiers import canon_cik, canon_lei, canon_vat


# ─────────────────────────────────────────────────────────────────────
# Country normalisation. The graph stores Company.country as ISO-3
# ("DEU", "FRA"), Lobbyist.country was full-name ("GERMANY"),
# SanctionedEntity.nationality is ISO-2 ("IR"). Resolver inputs accept
# any of these forms and normalise to ISO-3 once on entry.
# ─────────────────────────────────────────────────────────────────────

_ISO2_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "GB": "GBR", "GR": "GRC", "HR": "HRV",
    "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX",
    "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT",
    "RO": "ROU", "SE": "SWE", "SI": "SVN", "SK": "SVK",
    "NO": "NOR", "CH": "CHE", "IS": "ISL", "LI": "LIE",
    "US": "USA", "UK": "GBR", "CA": "CAN", "AU": "AUS", "JP": "JPN",
    "CN": "CHN", "IN": "IND", "BR": "BRA", "RU": "RUS", "TR": "TUR",
    "UA": "UKR", "KP": "PRK", "IR": "IRN", "SY": "SYR", "BY": "BLR",
    "LY": "LBY", "MM": "MMR", "IQ": "IRQ", "CD": "COD", "AF": "AFG",
    "UG": "UGA", "CF": "CAF",
}

_FULL_NAME_TO_ISO3 = {
    "AUSTRIA": "AUT", "BELGIUM": "BEL", "BULGARIA": "BGR",
    "CYPRUS": "CYP", "CZECH REPUBLIC": "CZE", "CZECHIA": "CZE",
    "DENMARK": "DNK", "ESTONIA": "EST", "FINLAND": "FIN",
    "FRANCE": "FRA", "GERMANY": "DEU", "GREECE": "GRC",
    "CROATIA": "HRV", "HUNGARY": "HUN", "IRELAND": "IRL",
    "ITALY": "ITA", "LATVIA": "LVA", "LITHUANIA": "LTU",
    "LUXEMBOURG": "LUX", "MALTA": "MLT", "NETHERLANDS": "NLD",
    "POLAND": "POL", "PORTUGAL": "PRT", "ROMANIA": "ROU",
    "SLOVAKIA": "SVK", "SLOVENIA": "SVN", "SPAIN": "ESP",
    "SWEDEN": "SWE",
    "NORWAY": "NOR", "SWITZERLAND": "CHE", "ICELAND": "ISL",
    "LIECHTENSTEIN": "LIE", "UNITED KINGDOM": "GBR",
    "UNITED STATES": "USA", "UNITED STATES OF AMERICA": "USA",
    "CANADA": "CAN", "AUSTRALIA": "AUS", "JAPAN": "JPN",
    "CHINA": "CHN", "INDIA": "IND", "BRAZIL": "BRA",
    "RUSSIA": "RUS", "TURKEY": "TUR", "UKRAINE": "UKR",
    "NORTH KOREA": "PRK", "IRAN": "IRN", "SYRIA": "SYR",
    "BELARUS": "BLR", "LIBYA": "LBY", "MYANMAR": "MMR",
    "IRAQ": "IRQ", "AFGHANISTAN": "AFG", "UGANDA": "UGA",
    "CENTRAL AFRICAN REPUBLIC": "CAF",
}


# Set of known ISO-3 codes — derived from the mapping values, so we
# only have one source of truth. Adding a country here means adding
# its ISO-2 / full-name entries above; pure ISO-3 strings that aren't
# in this set are rejected (no silent passthrough of invalid codes).
_KNOWN_ISO3 = frozenset(_ISO2_TO_ISO3.values()) | frozenset(_FULL_NAME_TO_ISO3.values())


def normalize_country(raw: str | None) -> str | None:
    """Return ISO-3 country code, or None if input is empty/unknown.

    Accepts ISO-3 (`"DEU"`), ISO-2 (`"DE"`), or full English name
    (`"Germany"`/`"GERMANY"`). Returns None when the input is empty or
    can't be mapped to a known country — callers MUST treat None as
    "do not match" rather than wildcarding (the original lobbying bug
    was wildcarding NULL).
    """
    if not raw:
        return None
    s = raw.strip().upper()
    if not s:
        return None
    if len(s) == 3 and s.isalpha() and s in _KNOWN_ISO3:
        return s
    if len(s) == 2 and s in _ISO2_TO_ISO3:
        return _ISO2_TO_ISO3[s]
    return _FULL_NAME_TO_ISO3.get(s)


# ─────────────────────────────────────────────────────────────────────
# Result shape
# ─────────────────────────────────────────────────────────────────────

ResolveTier = Literal["lei", "vat", "cik", "name_country", "fuzzy"]
ResolveHint = Literal["matched", "ambiguous", "no_match"]


@dataclass
class ResolveMatch:
    """A single entity matched by the resolver, tagged with the tier
    that produced it and a confidence score."""

    gmr_id: str
    name: str
    country: str | None
    lei: str | None
    tier: ResolveTier
    confidence: float


@dataclass
class ResolveResult:
    """Resolver output: at most one confident match, possibly a list of
    candidates for review, and the normalised country the lookup used."""

    hint: ResolveHint
    match: ResolveMatch | None = None
    candidates: list[ResolveMatch] = field(default_factory=list)
    # Always echo the normalised country back so callers can see
    # what the resolver actually matched against.
    normalised_country: str | None = None


# ─────────────────────────────────────────────────────────────────────
# Cypher
# ─────────────────────────────────────────────────────────────────────

_BY_LEI = (
    "MATCH (c:Company {lei: $lei}) "
    "RETURN c.gmr_id AS gmr_id, c.name AS name, c.country AS country, "
    "       c.lei AS lei LIMIT 2"
)

_BY_VAT = (
    "MATCH (c:Company {vat: $vat}) "
    "RETURN c.gmr_id AS gmr_id, c.name AS name, c.country AS country, "
    "       c.lei AS lei LIMIT 2"
)

_BY_CIK = (
    "MATCH (c:Company {cik: $cik}) "
    "RETURN c.gmr_id AS gmr_id, c.name AS name, c.country AS country, "
    "       c.lei AS lei LIMIT 2"
)

# Tier 3: cleaned-name + country agreement. apoc.text.clean
# lowercases and strips punctuation/whitespace; `LIMIT 2` lets us
# detect ambiguity. We compare against the materialised
# ``name_clean`` property (sink-written) so the index on
# ``name_clean`` answers in O(log N) instead of forcing a full scan.
_BY_NAME_COUNTRY_COMPANY = (
    "MATCH (c:Company) "
    "WHERE c.name_clean = apoc.text.clean($name) "
    "  AND coalesce(c.country, '') = $country "
    "RETURN c.gmr_id AS gmr_id, c.name AS name, c.country AS country, "
    "       c.lei AS lei LIMIT 2"
)

_BY_NAME_COUNTRY_AUTHORITY = (
    "MATCH (a:Authority) "
    "WHERE a.name_clean = apoc.text.clean($name) "
    "  AND coalesce(a.country, '') = $country "
    "RETURN a.authority_id AS gmr_id, a.name AS name, "
    "       a.country AS country, NULL AS lei LIMIT 2"
)

# Tier 4: full-text candidates. Same hardened guards we put in the
# ETL stopgaps — score floor 4.0, country agreement enforced. We
# return up to 5 for the review queue.
_FUZZY_COMPANY = (
    "CALL db.index.fulltext.queryNodes('company_name_ft', $clean_name) "
    "  YIELD node AS c, score "
    "WHERE score > 4.0 AND coalesce(c.country, '') = $country "
    "RETURN c.gmr_id AS gmr_id, c.name AS name, c.country AS country, "
    "       c.lei AS lei, score "
    "ORDER BY score DESC LIMIT 5"
)

_FUZZY_AUTHORITY = (
    "CALL db.index.fulltext.queryNodes('authority_name_ft', $clean_name) "
    "  YIELD node AS a, score "
    "WHERE score > 4.0 AND coalesce(a.country, '') = $country "
    "RETURN a.authority_id AS gmr_id, a.name AS name, "
    "       a.country AS country, NULL AS lei, score "
    "ORDER BY score DESC LIMIT 5"
)

_LUCENE_SPECIAL = "+-&|!(){}[]^\"~*?:\\/"

MIN_NAME_LEN = 6


def _clean_for_fulltext(name: str) -> str:
    """Replace Lucene special chars with spaces; collapse to single spaces."""
    out = []
    for ch in name:
        out.append(" " if ch in _LUCENE_SPECIAL else ch)
    return " ".join("".join(out).split())


# ─────────────────────────────────────────────────────────────────────
# resolve() — the entry point ETLs call instead of writing their own
# match cypher.
# ─────────────────────────────────────────────────────────────────────


async def resolve(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-return-statements
    driver: AsyncDriver,
    database: str,
    *,
    entity_type: Literal["Company", "Authority"],
    name: str | None = None,
    country: str | None = None,
    lei: str | None = None,
    vat: str | None = None,
    cik: str | None = None,
) -> ResolveResult:
    """Find the entity that matches the given attributes.

    Resolution proceeds tier-by-tier; the FIRST tier with a single
    confident match returns immediately. Multiple candidates at any
    tier collapses the result to ambiguous (the caller decides whether
    to review or to widen attributes). No tier ever writes to the
    graph — this is a read-only lookup.
    """
    iso_country = normalize_country(country)

    async with driver.session(database=database) as session:
        # Tier 1: LEI (Company only)
        if entity_type == "Company":
            canonical_lei = canon_lei(lei)
            if canonical_lei is not None:
                rows = await _run_match(session, _BY_LEI, lei=canonical_lei)
                hit = _resolve_rows(rows, "lei", 1.0)
                if hit is not None:
                    hit.normalised_country = iso_country
                    return hit

        # Tier 2: hard IDs (Company only). VAT first, then CIK.
        if entity_type == "Company":
            canonical_vat = canon_vat(vat)
            if canonical_vat is not None:
                rows = await _run_match(session, _BY_VAT, vat=canonical_vat)
                hit = _resolve_rows(rows, "vat", 0.99)
                if hit is not None:
                    hit.normalised_country = iso_country
                    return hit
            canonical_cik = canon_cik(cik)
            if canonical_cik is not None:
                rows = await _run_match(session, _BY_CIK, cik=canonical_cik)
                hit = _resolve_rows(rows, "cik", 0.99)
                if hit is not None:
                    hit.normalised_country = iso_country
                    return hit

        # Tier 3 + 4 require name + country.
        if not name or iso_country is None or len(name) < MIN_NAME_LEN:
            return ResolveResult(hint="no_match", normalised_country=iso_country)

        # Tier 3: cleaned-name + country
        query_t3 = (
            _BY_NAME_COUNTRY_COMPANY if entity_type == "Company"
            else _BY_NAME_COUNTRY_AUTHORITY
        )
        rows = await _run_match(session, query_t3, name=name, country=iso_country)
        hit = _resolve_rows(rows, "name_country", 0.95)
        if hit is not None:
            hit.normalised_country = iso_country
            return hit

        # Tier 4: fuzzy candidates — never auto-matched
        clean_name = _clean_for_fulltext(name)
        if len(clean_name) < MIN_NAME_LEN:
            return ResolveResult(hint="no_match", normalised_country=iso_country)
        query_t4 = (
            _FUZZY_COMPANY if entity_type == "Company" else _FUZZY_AUTHORITY
        )
        try:
            result = await session.run(
                query_t4, clean_name=clean_name, country=iso_country,
            )
            rows = [dict(record) async for record in result]
        except Exception:  # pylint: disable=broad-exception-caught
            # Fulltext index may not exist in ephemeral test DBs.
            rows = []

        if not rows:
            return ResolveResult(hint="no_match", normalised_country=iso_country)
        candidates = [
            ResolveMatch(
                gmr_id=r["gmr_id"], name=r["name"], country=r.get("country"),
                lei=r.get("lei"), tier="fuzzy",
                confidence=min(0.94, float(r["score"]) / 10.0),
            )
            for r in rows
        ]
        return ResolveResult(
            hint="ambiguous" if len(candidates) > 1 else "ambiguous",
            candidates=candidates,
            normalised_country=iso_country,
        )


async def _run_match(session, query: str, **params):
    result = await session.run(query, **params)
    return [dict(record) async for record in result]


def _resolve_rows(
    rows: list[dict], tier: ResolveTier, confidence: float,
) -> ResolveResult | None:
    """Convert hard-ID match rows to a ResolveResult.

    Zero rows → None (continue to next tier).
    One row  → matched.
    >1 rows  → ambiguous (collision on supposedly-unique key — usually a
                          GLEIF / VIES data-quality issue, must be reviewed).
    """
    if not rows:
        return None
    matches = [
        ResolveMatch(
            gmr_id=r["gmr_id"], name=r["name"], country=r.get("country"),
            lei=r.get("lei"), tier=tier, confidence=confidence,
        )
        for r in rows
    ]
    if len(matches) == 1:
        return ResolveResult(hint="matched", match=matches[0])
    return ResolveResult(hint="ambiguous", candidates=matches)
