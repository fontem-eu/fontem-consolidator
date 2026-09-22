"""GET /fiscal-id: do we hold this number, and under which prefix?

The failure this exists for (data-backlog C3): eForms `cbc:CompanyID`
carries a bare PT NIF, canon_vat returns None on it, the matcher name-
matches and mints a duplicate. The endpoint has to answer for the bare
number with a country, for the bare number alone (every prefix it
fits), for a number that already carries its prefix, and for Greece,
which spells its prefix two ways.
"""
# pylint: disable=protected-access,unused-argument
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator import fiscal_id as fid
from src.consolidator import identifiers as I
from src.consolidator.neo4j.migrations import INDEX_CYPHER

# ── identifiers: the prefix table and the VAT forms ──────────────────


@pytest.mark.parametrize("iso3,prefixes", [
    ("PRT", ("PT",)),
    ("prt", ("PT",)),
    ("GRC", ("EL", "GR")),   # VIES files Greece as EL; other sources carry GR
    ("GBR", ("GB",)),
    ("CHE", ("CH",)),        # a prefix, but canon_vat has no pattern for it
    ("USA", ()),
    (None, ()),
])
def test_vat_prefixes_by_iso3(iso3, prefixes):
    assert I.vat_prefixes(iso3) == prefixes


def test_split_vat_separates_prefix_from_national_part():
    assert I.split_vat("PT503536717") == ("PT", "503536717")
    assert I.split_vat("pt 503.536-717") == ("PT", "503536717")
    assert I.split_vat("ATU12345678") == ("AT", "U12345678")
    assert I.split_vat("503536717") is None
    assert I.split_vat(None) is None


def test_vat_forms_are_the_canonical_spellings_for_the_country():
    assert I.vat_forms("PRT", "503536717") == ["PT503536717"]
    assert I.vat_forms("GRC", "123456789") == ["EL123456789", "GR123456789"]
    # Not VAT-shaped in that country: nothing, the lookup runs on the
    # registry number alone.
    assert not I.vat_forms("DEU", "HRB117457")
    assert not I.vat_forms("CHE", "123456789")


def test_normalise_id_strips_punctuation_and_case():
    assert I.normalise_id(" pt-503.536/717 ") == "PT503536717"
    assert I.normalise_id("-- . /") == ""
    assert I.normalise_id(None) == ""


def test_known_vat_prefixes_include_both_greek_spellings():
    prefixes = I.known_vat_prefixes()
    assert "EL" in prefixes and "GR" in prefixes and "PT" in prefixes


# ── hypotheses ───────────────────────────────────────────────────────


def _countries(hyps):
    return [h.country for h in hyps]


def test_prefixed_number_is_its_own_hypothesis():
    hyps = fid.hypotheses("PT503536717")
    assert len(hyps) == 1
    assert hyps[0] == fid.Hypothesis("PRT", "503536717", ("PT503536717",))


def test_bare_number_with_country_is_one_hypothesis():
    for country in ("PT", "PRT", "pt", "Portugal"):
        hyps = fid.hypotheses("503 536 717", country)
        assert hyps == [fid.Hypothesis("PRT", "503536717", ("PT503536717",))], country


def test_bare_number_without_country_fans_out_to_every_prefix_it_fits():
    """A 9-digit number is a well-formed VAT in a dozen countries; the
    caller sees all of them rather than a guess."""
    hyps = fid.hypotheses("503536717")
    countries = _countries(hyps)
    for expected in ("PRT", "DEU", "GRC", "GBR", "ESP", "BEL"):
        assert expected in countries
    assert "ITA" not in countries  # IT wants 11 digits
    assert "FRA" not in countries  # FR wants a 2-char key + 9 digits
    assert len(countries) == len(set(countries)), "each country once"
    assert all(h.number == "503536717" for h in hyps)


def test_greece_is_one_hypothesis_with_two_vat_forms():
    for country in ("GR", "EL", "GRC"):
        hyps = fid.hypotheses("123456789", country)
        assert hyps == [fid.Hypothesis("GRC", "123456789", ("EL123456789", "GR123456789"))]
    # A GR-prefixed number is looked up under EL as well, and vice versa.
    assert fid.hypotheses("GR123456789") == fid.hypotheses("EL123456789")
    assert fid.hypotheses("GR123456789")[0].vat_forms == ("EL123456789", "GR123456789")


def test_prefixed_number_and_a_different_country_are_both_hypotheses():
    """The number says DE, the caller says PT: both are checked and the
    response shows both rather than silently picking one."""
    assert _countries(fid.hypotheses("DE123456789", "PT")) == ["DEU", "PRT"]


def test_country_without_a_vat_pattern_still_yields_a_hypothesis():
    hyps = fid.hypotheses("123456789", "CH")
    assert hyps == [fid.Hypothesis("CHE", "123456789", ())]


def test_non_vat_shaped_number_with_country_runs_on_the_registry_alone():
    hyps = fid.hypotheses("HRB 117457", "DE")
    assert hyps == [fid.Hypothesis("DEU", "HRB117457", ())]


def test_nothing_to_ask():
    assert fid.hypotheses("- . /") == []
    assert fid.hypotheses("") == []
    # bare, no country, fits no prefix
    assert fid.hypotheses("HRB117457") == []


# ── check(): the graph questions ─────────────────────────────────────


def _mk_session(rows_by_query: dict[str, list[dict]]):
    """A session whose .run() answers by query substring; unmatched
    queries return no rows. Records every (query, params)."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.calls = []

    async def _run(query, **kwargs):
        session.calls.append((query, kwargs))
        rows = []
        for needle, r in rows_by_query.items():
            if needle in query:
                rows = r
                break

        async def _aiter():
            for r in rows:
                yield r

        result = MagicMock()
        result.__aiter__ = lambda self: _aiter()
        return result

    session.run = AsyncMock(side_effect=_run)
    return session


def _driver(session):
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver


def _check(session, number, country=None):
    return asyncio.run(fid.check(_driver(session), "neo4j", number=number, country=country))


def _row(gmr_id, name):
    return {"gmr_id": gmr_id, "name": name, "country": "PRT", "lei": None}


def test_held_under_the_prefix_via_vat():
    session = _mk_session({"{vat: $vat}": [_row("vf-1", "Visualforma, S.A.")]})
    out = _check(session, "503536717", "PT")
    assert out["held"] is True
    assert out["query"] == {"number": "503536717", "country": "PRT"}
    assert out["candidates"] == [{
        "vat": "PT503536717", "country": "PRT", "matched": True,
        "entity_type": "Company", "gmr_id": "vf-1", "authority_id": None,
        "name": "Visualforma, S.A.", "matched_on": "vat",
    }]
    vat_call = next(kw for q, kw in session.calls if "{vat: $vat}" in q)
    assert vat_call["vat"] == "PT503536717"


def test_held_via_registered_as_with_the_bare_number_and_country():
    session = _mk_session({"registered_as: $registered_as": [_row("vf-2", "Visualforma")]})
    out = _check(session, "PT503536717")
    assert out["held"] is True
    cand = out["candidates"][0]
    assert cand["matched_on"] == "registered_as"
    assert cand["vat"] == "PT503536717", "reported under the VIES spelling"
    reg_call = next(kw for q, kw in session.calls if "registered_as: $registered_as" in q)
    assert reg_call == {"registered_as": "503536717", "country": "PRT", "limit": 10}


def test_held_by_an_authority_under_either_spelling():
    session = _mk_session({"a.national_id IN $ids": [_row("AUTH-9", "Município de Beja")]})
    out = _check(session, "503536717", "PT")
    cand = out["candidates"][0]
    assert cand["entity_type"] == "Authority"
    assert cand["authority_id"] == "AUTH-9" and cand["gmr_id"] is None
    assert cand["matched_on"] == "national_id"
    auth_call = next(kw for q, kw in session.calls if "a.national_id IN $ids" in q)
    assert auth_call["ids"] == ["PT503536717", "503536717"]
    assert auth_call["country"] == "PRT"


def test_a_node_holding_the_number_twice_is_listed_once():
    """vat and registered_as both hit the same node: one row, under vat."""
    session = _mk_session({
        "{vat: $vat}": [_row("vf-1", "Visualforma")],
        "registered_as: $registered_as": [_row("vf-1", "Visualforma"), _row("vf-3", "Other")],
    })
    out = _check(session, "503536717", "PT")
    assert [(c["gmr_id"], c["matched_on"]) for c in out["candidates"]] == [
        ("vf-1", "vat"), ("vf-3", "registered_as"),
    ]


def test_not_held_is_one_unmatched_row_per_hypothesis():
    session = _mk_session({})
    out = _check(session, "503536717")
    assert out["held"] is False
    assert out["query"]["country"] is None
    assert all(c["matched"] is False for c in out["candidates"])
    assert all(c["entity_type"] is None and c["gmr_id"] is None for c in out["candidates"])
    by_country = {c["country"]: c["vat"] for c in out["candidates"]}
    assert by_country["PRT"] == "PT503536717"
    assert by_country["GRC"] == "EL123456789".replace("123456789", "503536717")


def test_greece_asks_for_both_spellings():
    session = _mk_session({"{vat: $vat}": [_row("gr-1", "ΟΤΕ")]})
    out = _check(session, "123456789", "GR")
    vats = [kw["vat"] for q, kw in session.calls if "{vat: $vat}" in q]
    assert vats == ["EL123456789", "GR123456789"]
    # the same node under both spellings is still one candidate
    assert [c["gmr_id"] for c in out["candidates"]] == ["gr-1"]
    assert out["candidates"][0]["vat"] == "EL123456789"


def test_a_country_with_no_vat_pattern_skips_the_vat_tier():
    session = _mk_session({})
    out = _check(session, "123456789", "CH")
    assert not any("{vat: $vat}" in q for q, _ in session.calls)
    assert any("registered_as" in q for q, _ in session.calls)
    assert out["candidates"] == [{
        "vat": None, "country": "CHE", "matched": False, "entity_type": None,
        "gmr_id": None, "authority_id": None, "name": None, "matched_on": None,
    }]


# ── the route ────────────────────────────────────────────────────────


def _route_driver(rows_by_query):
    session = _mk_session(rows_by_query)
    return _driver(session)


def test_route_returns_the_contract_shape(client):
    c, _ = client
    driver = _route_driver({"{vat: $vat}": [_row("vf-1", "Visualforma, S.A.")]})
    with patch("src.api.routes.fiscal_id.get_driver", AsyncMock(return_value=driver)):
        r = c.get("/fiscal-id/PT 503.536.717")
    assert r.status_code == 200
    body = r.json()
    assert body["held"] is True
    assert body["query"] == {"number": "PT503536717", "country": None}
    assert body["candidates"] == [{
        "vat": "PT503536717", "country": "PRT", "matched": True,
        "entity_type": "Company", "gmr_id": "vf-1", "authority_id": None,
        "name": "Visualforma, S.A.", "matched_on": "vat",
    }]


def test_route_bare_number_with_country(client):
    c, _ = client
    driver = _route_driver({})
    with patch("src.api.routes.fiscal_id.get_driver", AsyncMock(return_value=driver)):
        r = c.get("/fiscal-id/503536717?country=PRT")
    body = r.json()
    assert body["held"] is False
    assert body["query"]["country"] == "PRT"
    assert [cand["vat"] for cand in body["candidates"]] == ["PT503536717"]


def test_route_bare_number_without_country_lists_every_candidate(client):
    c, _ = client
    driver = _route_driver({})
    with patch("src.api.routes.fiscal_id.get_driver", AsyncMock(return_value=driver)):
        r = c.get("/fiscal-id/503536717")
    countries = {cand["country"] for cand in r.json()["candidates"]}
    assert {"PRT", "DEU", "GRC"} <= countries


def test_route_rejects_a_number_that_normalises_to_nothing(client):
    c, _ = client
    r = c.get("/fiscal-id/-.-")
    assert r.status_code == 400
    assert "nothing" in r.json()["detail"]


def test_route_rejects_an_unknown_country(client):
    c, _ = client
    r = c.get("/fiscal-id/503536717?country=Atlantis")
    assert r.status_code == 400
    assert "Atlantis" in r.json()["detail"]


# ── the index the Authority lookup relies on ─────────────────────────


def test_authority_national_id_lookup_is_indexed():
    """Up to a dozen hypotheses per call, each asking Authority by
    national_id + country: a label scan per hypothesis would make the
    endpoint unusable for the cleaner."""
    assert any(
        "(a:Authority)" in stmt and "a.national_id" in stmt and "a.country" in stmt
        for stmt in INDEX_CYPHER
    )
