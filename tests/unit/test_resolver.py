"""Unit tests for the entity-identification resolver.

The resolver is the single entry point ETLs will use to find an
existing graph entity from a bag of attributes. These tests cover:
  - country normalisation (ISO-2/ISO-3/full name → ISO-3)
  - tier-by-tier resolution (LEI > VAT > CIK > name+country > fuzzy)
  - the historical ETL false-positive cases (sanctions, lobbying)
    must NOT come back as confident matches under the new logic
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.consolidator import resolver
from src.consolidator.resolver import (
    MIN_NAME_LEN,
    ResolveMatch,
    ResolveResult,
    _resolve_rows,
    normalize_country,
)


# ─────────────────────────────────────────────────────────────────────
# Country normalisation — the missing primitive that broke the
# lobbying matcher in production.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    # Already ISO-3
    ("DEU", "DEU"),
    ("FRA", "FRA"),
    ("deu", "DEU"),  # lowercase OK
    # ISO-2
    ("DE", "DEU"),
    ("FR", "FRA"),
    ("US", "USA"),
    ("UK", "GBR"),  # informal alias
    # Full name
    ("Germany", "DEU"),
    ("GERMANY", "DEU"),
    ("Czech Republic", "CZE"),
    ("Czechia", "CZE"),
    ("United Kingdom", "GBR"),
    ("United States of America", "USA"),
    # Sanctions-list nationalities (ISO-2)
    ("IR", "IRN"),
    ("KP", "PRK"),
    ("UG", "UGA"),
])
def test_normalize_country_known(raw, expected):
    assert normalize_country(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "ZZ", "ZZZ", "Atlantis"])
def test_normalize_country_returns_none_for_unknown(raw):
    """Unknown / empty must return None — callers MUST treat None as
    'do not match' rather than wildcarding (the original lobbying bug)."""
    assert normalize_country(raw) is None


# ─────────────────────────────────────────────────────────────────────
# _resolve_rows — converts hard-ID match rows to a ResolveResult.
# Single row → matched, multiple → ambiguous (collision), zero → None.
# ─────────────────────────────────────────────────────────────────────


def _row(gmr_id="A", name="Acme SAS", country="FRA", lei="X" * 20):
    return {"gmr_id": gmr_id, "name": name, "country": country, "lei": lei}


def test_resolve_rows_single_returns_matched():
    result = _resolve_rows([_row()], "lei", 1.0)
    assert result.hint == "matched"
    assert result.match is not None
    assert result.match.tier == "lei"
    assert result.match.confidence == 1.0


def test_resolve_rows_multiple_returns_ambiguous():
    """Two matches on a supposedly-unique key (LEI/VAT) is a data-quality
    incident — usually a GLEIF dupe — and must surface as ambiguous so
    the caller can pick by another signal or queue for review."""
    result = _resolve_rows([_row("A"), _row("B")], "lei", 1.0)
    assert result.hint == "ambiguous"
    assert result.match is None
    assert len(result.candidates) == 2


def test_resolve_rows_zero_returns_none():
    """Zero rows means: nothing at this tier, try the next tier."""
    assert _resolve_rows([], "lei", 1.0) is None


# ─────────────────────────────────────────────────────────────────────
# Tier ordering — LEI wins over VAT wins over CIK wins over name+country.
# Verified by checking which cypher session.run was called with.
# ─────────────────────────────────────────────────────────────────────


def _mk_session(rows_by_query: dict[str, list[dict]]):
    """Build a mock session whose .run() returns rows keyed by query
    substring. Anything not matched returns []."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    async def _run(query, **_kwargs):
        result = MagicMock()
        rows = []
        for needle, r in rows_by_query.items():
            if needle in query:
                rows = r
                break

        async def _aiter():
            for r in rows:
                yield r

        result.__aiter__ = lambda self: _aiter()
        return result

    session.run = AsyncMock(side_effect=_run)
    return session


def _mk_driver(session):
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver


@pytest.mark.asyncio
async def test_lei_wins_when_present():
    session = _mk_session({"MATCH (c:Company {lei:": [_row(gmr_id="lei-hit")]})
    result = await resolver.resolve(
        _mk_driver(session), "neo4j",
        entity_type="Company",
        lei="X" * 20, name="Should Be Ignored", country="FR",
    )
    assert result.hint == "matched"
    assert result.match.tier == "lei"
    assert result.match.gmr_id == "lei-hit"


@pytest.mark.asyncio
async def test_vat_used_when_lei_absent():
    session = _mk_session({"MATCH (c:Company {vat:": [_row(gmr_id="vat-hit")]})
    result = await resolver.resolve(
        _mk_driver(session), "neo4j",
        entity_type="Company", vat="DE123456789", country="DE",
    )
    assert result.hint == "matched"
    assert result.match.tier == "vat"


@pytest.mark.asyncio
async def test_no_match_when_no_attrs_workable():
    session = _mk_session({})
    result = await resolver.resolve(
        _mk_driver(session), "neo4j",
        entity_type="Company", name="X",  # too short
    )
    assert result.hint == "no_match"


# ─────────────────────────────────────────────────────────────────────
# Country normalisation must drive the tier-3 query.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_name_country_uses_normalised_iso3():
    captured = []

    async def _run(query, **kwargs):
        captured.append({"query": query, "params": kwargs})
        result = MagicMock()

        async def _aiter():
            return
            yield  # pragma: no cover

        result.__aiter__ = lambda self: _aiter()
        return result

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.run = AsyncMock(side_effect=_run)
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)

    await resolver.resolve(
        driver, "neo4j",
        entity_type="Company",
        name="Adyen N.V. Long Enough",
        country="Netherlands",  # full name; should normalise to NLD
    )
    name_country_calls = [
        c for c in captured if "apoc.text.clean" in c["query"]
    ]
    assert name_country_calls, "expected a tier-3 cypher call"
    # All tier-3 / tier-4 calls receive the normalised country
    for call in name_country_calls:
        assert call["params"].get("country") == "NLD"


# ─────────────────────────────────────────────────────────────────────
# Historical false positives — none of these inputs may yield a
# confident match. Same case set as the ETL test suites.
# ─────────────────────────────────────────────────────────────────────

HISTORICAL_FALSE_POSITIVE_CASES = [
    pytest.param(
        {"name": "AMD", "country": "FR"},
        id="AMD-too-short",
    ),
    pytest.param(
        {"name": "TSA", "country": "DK"},
        id="TSA-too-short",
    ),
    pytest.param(
        {"name": "LRA", "country": "FR"},
        id="LRA-too-short",
    ),
    pytest.param(
        {"name": "NADA", "country": "BE"},
        id="NADA-too-short",
    ),
    pytest.param(
        {"name": "CRL", "country": "FR"},
        id="CRL-too-short",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("attrs", HISTORICAL_FALSE_POSITIVE_CASES)
async def test_short_acronym_rejected_at_tier3(attrs):
    """The MIN_NAME_LEN guard rejects all the acronym false-positive
    cases before any cypher even runs. Without LEI/VAT/CIK there is
    nothing to match against."""
    session = _mk_session({})
    result = await resolver.resolve(
        _mk_driver(session), "neo4j", entity_type="Company", **attrs,
    )
    assert result.hint == "no_match"
    # Verify no name-based query was attempted
    for call in session.run.call_args_list:
        query = call.args[0]
        assert "apoc.text.clean" not in query, (
            f"name-based cypher must not run for short name {attrs['name']!r}"
        )
        assert "queryNodes" not in query


def test_min_name_len_at_least_six():
    assert MIN_NAME_LEN >= 6


# ─────────────────────────────────────────────────────────────────────
# Dataclass smoke
# ─────────────────────────────────────────────────────────────────────


def test_resolve_result_default_candidates_is_empty_list():
    """Default factory must produce an empty list (not shared default)."""
    a = ResolveResult(hint="no_match")
    b = ResolveResult(hint="no_match")
    assert a.candidates == []
    a.candidates.append(
        ResolveMatch(gmr_id="x", name="x", country="DEU", lei=None,
                     tier="lei", confidence=1.0)
    )
    assert b.candidates == [], "candidates must not be shared between instances"
