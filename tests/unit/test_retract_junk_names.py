"""retract_junk_names: withdraw every equivalence a junk-named company
took part in, emit first and record second, report what was touched.

What this pins: the patterns find the Italian family and nothing
legitimate; the scan pre-filter and the Python decision agree; a dry
run writes and emits nothing; a run batches, records only after the
batch landed, and stops on a short batch; the report carries the
contracts and pairs the node-removal step will need; a pair already
corrected is not re-emitted.
"""
# pylint: disable=protected-access,unused-argument,attribute-defined-outside-init
from __future__ import annotations

import asyncio
import io
import json
import re
import sys
from unittest.mock import AsyncMock, patch

import pytest

from src.consolidator import retract_junk_names as rj

# ── patterns ──────────────────────────────────────────────────────────

JUNK = [
    "Gara aggiudicata come da determina n. 543 del 2013 pubblicata sul sito www.csc.sanita.fvg.it",
    "GARA AGGIUDICATA COME DA DETERMINA N. 12 DEL 2011",
    "Si veda determina n. 7 del 2012",
    "vedi allegato",
    "Aggiudicato con determina dirigenziale",
    "Elenco pubblicato sul sito internet",
    "Fornitore: http://example.it/albo",
    "Ditta www.example.it",
]
LEGIT = [
    "Visualforma - Tecnologias de Informação, S.A.",
    "Siemens Aktiengesellschaft",
    "Vediamo S.r.l.",
    # Found by the first scan on shared: real names a bare `vedi ` hit.
    "BAN, S. SZABO & PARTNERS UGYVEDI IRODA",
    "OPTIMUM FUND - CSOB BYCI A MEDVEDI PLUS 1",
    "Sivedali S.p.A.",
]
#: The plan's language-agnostic hard signals are blunt on purpose; the
#: dry run and the country filter are what keep them from real names.
BLUNT = ["Httpool Ltd", "WWW.THETAXSHOP.COM LIMITED"]


@pytest.mark.parametrize("name", JUNK)
def test_the_default_patterns_catch_the_italian_family(name):
    assert rj.matched_patterns(name, rj.compile_patterns(list(rj.DEFAULT_PATTERNS)))


@pytest.mark.parametrize("name", LEGIT)
def test_the_default_patterns_leave_real_names_alone(name):
    assert not rj.matched_patterns(name, rj.compile_patterns(list(rj.DEFAULT_PATTERNS)))


@pytest.mark.parametrize("name", BLUNT)
def test_www_and_http_are_blunt_signals_by_design(name):
    hit = rj.matched_patterns(name, rj.compile_patterns(list(rj.DEFAULT_PATTERNS)))
    assert hit and all(p in (r"(?i)www\.", r"(?i)http") for p in hit)


def test_matched_patterns_reports_every_pattern_that_hit_in_order():
    compiled = rj.compile_patterns(list(rj.DEFAULT_PATTERNS))
    hit = rj.matched_patterns(JUNK[0], compiled)
    assert hit == [r"(?i)^gara aggiudicata", r"(?i)\bcome da determina\b",
                   r"(?i)\bdetermina n\.", r"(?i)\bpubblicat[ao] sul sito\b",
                   r"(?i)www\."]
    assert rj.matched_patterns(None, compiled) == []
    assert rj.matched_patterns("", compiled) == []


def test_cypher_regex_hoists_flags_and_wraps_unanchored_patterns():
    rx = rj.cypher_regex([r"(?i)^gara aggiudicata", r"(?i)www\.", r"http"])
    assert rx == r"(?is)(?:(?:^gara aggiudicata).*|.*(?:www\.).*|.*(?:http).*)"


@pytest.mark.parametrize("name", JUNK + LEGIT + BLUNT)
def test_the_scan_prefilter_agrees_with_the_decision(name):
    """The combined regex is a Java full-match on the Neo4j side; for the
    syntax these patterns use Python's fullmatch behaves the same, so a
    name the decision matches is one the scan returns."""
    patterns = list(rj.DEFAULT_PATTERNS)
    decided = bool(rj.matched_patterns(name, rj.compile_patterns(patterns)))
    scanned = re.fullmatch(rj.cypher_regex(patterns), name) is not None
    assert scanned == decided, name


SPARED = [
    # Found by the scoped scan on shared: Italian companies a bare
    # `vedi` / `www.` reached. Each carries a legal form.
    "GRUPPO VEDI S.R.L.",
    "VEDI VISION S.R.L.",
    "VEDI VISION - SOCIETA' A RESPONSABILITA' LIMITATA",
    "AN VEDI DUE SOCIETA' A RESPONSABILITA' LIMITATA",
    "WWW.ROBINSONPETSHOP.IT SRL",
    "RTI Vedi Vision Srl – Fimas Srl",
    "WWW.THETAXSHOP.COM LIMITED",
    "www.levnedobijeni.cz s.r.o.",
    "WWW.IUSLABORIS.PL SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ".replace(
        "SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ", "SP. Z O.O."),
    "DOMIX www.dobreokna.net M.Różycki Sp.j.".replace("Sp.j.", "GmbH"),
]


@pytest.mark.parametrize("name", SPARED)
def test_a_legal_form_marks_a_company_however_the_rest_reads(name):
    assert rj.skip_reason(name, False) == "legal_form"


@pytest.mark.parametrize(
    "name", JUNK + ["VEDI ATTI DI AGGIUDICAZIONE", "https://montedoro.traspare.com/."],
)
def test_notice_text_carries_no_legal_form(name):
    assert rj.skip_reason(name, False) is None


def test_a_hard_identifier_spares_the_node_before_the_name_is_read():
    assert rj.skip_reason(JUNK[0], True) == "hard_id"
    assert rj.skip_reason(None, False) is None


def test_pattern_file_replaces_the_defaults(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# junk\n(?i)^lotto n\\.\n\n  (?i)n/a  \n", encoding="utf-8")
    assert rj.load_patterns(str(f)) == [r"(?i)^lotto n\.", r"(?i)n/a"]
    assert rj.load_patterns(None) == list(rj.DEFAULT_PATTERNS)


# ── an in-memory graph that answers the command's three queries ───────


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        self._it = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def single(self):
        return self._rows[0] if self._rows else None


class _Graph:
    """Junk nodes, their partners, their contracts. Records every write
    statement (the correction Cypher) so tests can see what landed."""

    def __init__(self, nodes, pairs=(), contracts=()):
        self.nodes = list(nodes)              # {gmr_id, name}
        self.pairs = list(pairs)              # (junk_id, other_id, method)
        self.contracts = list(contracts)      # (junk_id, ted_notice_id)
        self.corrected: set[frozenset] = set()
        self.writes: list[tuple[str, dict]] = []
        self.scans = 0

    def run(self, query, **params):
        if "c.name =~ $regex" in query:
            self.scans += 1
            self.scan_query = query
            country = params.get("country") if "$country" in query else None
            return _Result([{
                "gmr_id": n["gmr_id"], "name": n["name"],
                "has_hard_id": bool(n.get("vat") or n.get("lei")),
                "same_as_degree": sum(n["gmr_id"] in p[:2] for p in self.pairs),
                "contract_degree": sum(c[0] == n["gmr_id"] for c in self.contracts),
            } for n in self.nodes if re.fullmatch(params["regex"], n["name"])
                and country in (None, n.get("country"))])
        if "SAME_AS|SAME_AS_CANDIDATE" in query:
            rows = []
            for junk, other, method in self.pairs:
                for me, you in ((junk, other), (other, junk)):
                    if me in params["ids"] and frozenset((junk, other)) not in self.corrected:
                        rows.append({"gmr_id": me, "other_id": you,
                                     "other_name": f"name of {you}", "method": method})
            return _Result(rows)
        if "AWARDED_TO" in query:
            return _Result([{"gmr_id": j, "ted_notice_id": t, "contract_key": f"k-{t}",
                             "title": f"title {t}"} for j, t in self.contracts
                            if j in params["ids"]])
        self.writes.append((query, params))
        if "NOT_SAME_AS" in query:
            self.corrected.add(frozenset((params["from"], params["to"])))
        return _Result([])


class _Driver:
    def __init__(self, graph):
        self.graph = graph

    def session(self, database=None):
        graph = self.graph

        class _S:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def run(self, query, **params):
                return graph.run(query, **params)
        return _S()


def _junk(i):
    return {"gmr_id": f"junk-{i}", "name": f"Gara aggiudicata come da determina n. {i} del 2011"}


def _scan_all(graph, patterns=None, country=None):
    return asyncio.run(rj.scan(
        _Driver(graph), "neo4j", patterns or list(rj.DEFAULT_PATTERNS), country,
    ))


def _scan(graph, patterns=None, country=None):
    return _scan_all(graph, patterns, country)[0]


# ── dry run ───────────────────────────────────────────────────────────


def test_a_dry_run_counts_per_pattern_shows_ten_examples_and_writes_nothing():
    graph = _Graph(
        [_junk(i) for i in range(12)] + [{"gmr_id": "ok", "name": "Siemens AG"},
                                         {"gmr_id": "web", "name": "Ditta www.x.it"}],
        pairs=[("junk-0", "junk-1", "fuzzy_name_same_country"),
               ("junk-0", "real-1", "fuzzy_name_same_country")],
        contracts=[("junk-0", "1-2013"), ("junk-1", "2-2013"), ("junk-1", "3-2013")],
    )
    emit = AsyncMock(return_value=0)
    with patch.object(rj.eventlog, "emit_retract_same_as_many", emit):
        nodes = _scan(graph)
        report = rj.dry_run_report(nodes, list(rj.DEFAULT_PATTERNS))
    assert report["dry_run"] is True and report["country"] is None
    assert report["counts"]["nodes"] == 13
    assert report["counts"]["skipped"] == {"legal_form": 0, "hard_id": 0}
    assert report["skipped_examples"] == []
    assert report["counts"]["by_pattern"][r"(?i)^gara aggiudicata"] == 12
    assert report["counts"]["by_pattern"][r"(?i)\bdetermina n\."] == 12
    assert report["counts"]["by_pattern"][r"(?i)www\."] == 1
    assert report["counts"]["by_pattern"][r"(?i)http"] == 0
    # degrees: junk-0 has 2 edges, junk-1 has 1; junk-0 1 contract, junk-1 2
    assert report["counts"]["same_as_edges"] == 3
    assert report["counts"]["contracts"] == 3
    assert len(report["examples"]) == rj.EXAMPLES == 10
    assert report["examples"][0] == {
        "gmr_id": "junk-0", "name": _junk(0)["name"],
        "patterns": [r"(?i)^gara aggiudicata", r"(?i)\bcome da determina\b",
                     r"(?i)\bdetermina n\."],
        "same_as_degree": 2, "contract_degree": 1,
    }
    assert not graph.writes and emit.await_count == 0


def test_the_scan_is_scoped_to_one_country_unless_lifted():
    """The defaults are Italian notice text; unscoped they reach real
    names elsewhere (a Hungarian law office, a Czech fund, on shared)."""
    graph = _Graph([{**_junk(0), "country": "ITA"}, {**_junk(1), "country": "HUN"},
                    {**_junk(2)}])
    assert [n["gmr_id"] for n in _scan(graph, country="ITA")] == ["junk-0"]
    assert "AND c.country = $country" in graph.scan_query
    assert [n["gmr_id"] for n in _scan(graph)] == ["junk-0", "junk-1", "junk-2"]
    assert "$country" not in graph.scan_query
    assert rj.scan_query(None).startswith("MATCH (c:Company) WHERE c.name =~ $regex\n")


def test_matched_companies_are_spared_and_shown():
    """The guard: a legal form or a hard identifier means a company, and
    a company's equivalences are not the tool's to retract."""
    graph = _Graph([
        {"gmr_id": "j", "name": "VEDI ATTI DI AGGIUDICAZIONE"},
        {"gmr_id": "s1", "name": "GRUPPO VEDI S.R.L."},
        {"gmr_id": "s2", "name": "WWW.ROBINSONPETSHOP.IT SRL"},
        {"gmr_id": "v", "name": _junk(0)["name"], "vat": "IT01234567890"},
    ], pairs=[("s1", "real", "exact_vat_match"), ("j", "junk-9", "fuzzy_name_same_country")])
    nodes, skipped = _scan_all(graph)
    assert [n["gmr_id"] for n in nodes] == ["j"]
    assert [(n["gmr_id"], n["skipped"]) for n in skipped] == [
        ("s1", "legal_form"), ("s2", "legal_form"), ("v", "hard_id"),
    ]
    assert skipped[0]["same_as_degree"] == 1 and skipped[0]["patterns"] == [r"(?i)\bvedi\b"]
    report = rj.dry_run_report(nodes, list(rj.DEFAULT_PATTERNS), "ITA", skipped)
    assert report["counts"] == {
        "nodes": 1, "by_pattern": {**{p: 0 for p in rj.DEFAULT_PATTERNS}, r"(?i)\bvedi\b": 1},
        "same_as_edges": 1, "contracts": 0, "skipped": {"legal_form": 2, "hard_id": 1},
    }
    assert [n["gmr_id"] for n in report["skipped_examples"]] == ["s1", "s2", "v"]
    # and a run touches only the junk node's pair
    (entries, retracted, _), emit = _run(graph, nodes, batch=10)
    assert retracted == 1 and [e["gmr_id"] for e in entries] == ["j"]
    assert emit.await_args_list[0].args[0][0]["b_iri"].endswith("/junk-9")


def test_the_scan_drops_rows_the_prefilter_over_matched():
    """The Cypher regex is case-insensitive for every pattern; a pattern
    without (?i) is decided case-sensitively in Python."""
    graph = _Graph([{"gmr_id": "a", "name": "VEDI ALLEGATO"},
                    {"gmr_id": "b", "name": "vedi allegato"}])
    assert [n["gmr_id"] for n in _scan(graph, [r"\bvedi\b"])] == ["b"]


# ── the run ───────────────────────────────────────────────────────────


def _run(graph, nodes, *, batch=2, emit=None):
    emit = emit or AsyncMock(side_effect=len)
    with patch.object(rj.eventlog, "emit_retract_same_as_many", emit):
        out = asyncio.run(rj.retract(
            _Driver(graph), "neo4j", nodes, batch=batch, reviewer="c1-bot",
        ))
    return out, emit


def test_a_run_emits_per_batch_then_records_each_pair():
    graph = _Graph(
        [_junk(i) for i in range(5)],
        pairs=[(f"junk-{i}", f"real-{i}", "fuzzy_name_same_country") for i in range(5)]
              + [("junk-0", "real-x", "exact_name_country_match")],
        contracts=[("junk-0", "1-2013"), ("junk-3", "9-2013")],
    )
    (entries, retracted, complete), emit = _run(graph, _scan(graph), batch=2)
    assert complete is True and retracted == 6
    assert emit.await_count == 3, "5 nodes in batches of 2 = 3 transactions"
    first_batch = emit.await_args_list[0].args[0]
    assert {(r["a_iri"], r["b_iri"]) for r in first_batch} == {
        ("http://data.fontem.eu/id/Company/junk-0", "http://data.fontem.eu/id/Company/real-0"),
        ("http://data.fontem.eu/id/Company/junk-0", "http://data.fontem.eu/id/Company/real-x"),
        ("http://data.fontem.eu/id/Company/junk-1", "http://data.fontem.eu/id/Company/real-1"),
    }
    assert first_batch[0]["reason"] == rj.REASON
    assert first_batch[0]["reviewer"] == "c1-bot"
    assert first_batch[0]["domain"] == "company"
    assert {r["retracted_method"] for r in first_batch} == {
        "fuzzy_name_same_country", "exact_name_country_match",
    }

    # Every pair got the endpoint's two writes: the correction and its log.
    corrections = [p for q, p in graph.writes if "NOT_SAME_AS" in q]
    logs = [p for q, p in graph.writes if "DecisionLog" in q]
    assert len(corrections) == len(logs) == 6
    assert {(c["from"], c["to"]) for c in corrections} >= {
        ("junk-0", "real-0"), ("junk-0", "real-x"),
    }
    assert all(c["reviewer"] == "c1-bot" and c["reason"] == rj.REASON for c in corrections)
    assert all(l["entity_type"] == "Company" and l["note"] == rj.REASON for l in logs)
    assert any("DELETE c" in q for q, _ in graph.writes), "the settled candidate must go"

    # The report: every node, its contracts, the pairs retracted.
    assert [e["gmr_id"] for e in entries] == [f"junk-{i}" for i in range(5)]
    assert entries[0]["contracts"] == [
        {"ted_notice_id": "1-2013", "contract_key": "k-1-2013", "title": "title 1-2013"},
    ]
    assert entries[1]["contracts"] == []
    assert entries[0]["same_as"] == [
        {"other_id": "real-0", "other_name": "name of real-0",
         "method": "fuzzy_name_same_country"},
        {"other_id": "real-x", "other_name": "name of real-x",
         "method": "exact_name_country_match"},
    ]
    assert all(e["retracted"] is True for e in entries)
    assert entries[0]["patterns"] and entries[0]["name"] == _junk(0)["name"]


def test_a_pair_between_two_junk_nodes_is_retracted_once():
    graph = _Graph([_junk(0), _junk(1)], pairs=[("junk-0", "junk-1", "fuzzy_name_same_country")])
    (entries, retracted, _), emit = _run(graph, _scan(graph), batch=10)
    assert retracted == 1 and emit.await_count == 1
    assert len(emit.await_args_list[0].args[0]) == 1
    # both nodes see the partner in their report entry
    assert entries[0]["same_as"][0]["other_id"] == "junk-1"
    assert entries[1]["same_as"][0]["other_id"] == "junk-0"


def test_a_short_batch_records_nothing_and_stops():
    """Emit first, record second, record nothing on a partial batch: a
    :NOT_SAME_AS behind a :SAME_AS the sink never deleted would look
    settled forever. The next run finds the pairs again."""
    graph = _Graph([_junk(i) for i in range(4)],
                   pairs=[(f"junk-{i}", f"real-{i}", "m") for i in range(4)])
    emit = AsyncMock(side_effect=[2, 0])
    (entries, retracted, complete), emit = _run(graph, _scan(graph), batch=2, emit=emit)
    assert complete is False
    assert retracted == 2 and emit.await_count == 2
    assert len([q for q, _ in graph.writes if "NOT_SAME_AS" in q]) == 2
    assert [e["retracted"] for e in entries] == [True, True, False, False]
    assert len(entries) == 4, "the failed batch is still reported"
    # a re-run finds exactly what did not land
    (_, again, _), _ = _run(graph, _scan(graph), batch=2)
    assert again == 2


def test_a_re_run_finds_nothing_to_do():
    graph = _Graph([_junk(0)], pairs=[("junk-0", "real-0", "m")])
    (_, first, _), _ = _run(graph, _scan(graph))
    (_, second, _), emit = _run(graph, _scan(graph))
    assert (first, second) == (1, 0)
    assert emit.await_count == 0, "nothing to emit means no transaction"


def test_the_read_query_skips_corrected_pairs_and_pending_proposals():
    """Idempotency and scope live in the Cypher: a pair with a
    :NOT_SAME_AS is done, and a pending candidate was never asserted so
    there is nothing to retract."""
    assert "NOT EXISTS { (c)-[:NOT_SAME_AS]-(o) }" in rj._PAIRS
    assert "type(e) = 'SAME_AS' OR e.status = 'approved'" in rj._PAIRS
    assert "o.gmr_id <> c.gmr_id" in rj._PAIRS


def test_run_report_shape():
    base = rj.dry_run_report([], ["p"])
    entries = [{"gmr_id": "j", "same_as": [{"other_id": "r"}], "contracts": [], "retracted": True}]
    report = rj.run_report(base, entries, 1, True)
    assert report["dry_run"] is False and report["complete"] is True
    assert report["examples"] is None and report["nodes"] == entries
    assert report["counts"] == {"nodes": 0, "by_pattern": {"p": 0}, "same_as_edges": 0,
                                "contracts": 0, "skipped": {"legal_form": 0, "hard_id": 0},
                                "pairs": 1, "retracted": 1}
    assert report["patterns"] == ["p"] and report["generated_at"]
    assert report["country"] is None
    assert rj.dry_run_report([], ["p"], "ITA")["country"] == "ITA"


# ── run() and the CLI ─────────────────────────────────────────────────


def test_run_writes_the_report_and_always_closes_the_driver():
    graph = _Graph([_junk(0)], pairs=[("junk-0", "real-0", "m")], contracts=[("junk-0", "1-2013")])
    out = io.StringIO()
    close = AsyncMock()
    with patch.object(rj, "get_driver", AsyncMock(return_value=_Driver(graph))), \
         patch.object(rj, "close_driver", close), \
         patch.object(rj.eventlog, "emit_retract_same_as_many", AsyncMock(side_effect=len)):
        complete = asyncio.run(rj.run(
            rj.Options(dry_run=False, batch=10, reviewer="c1-bot", country=None), out,
        ))
    close.assert_awaited_once()
    assert complete is True
    report = json.loads(out.getvalue())
    assert report["counts"]["retracted"] == 1
    assert report["nodes"][0]["contracts"][0]["ted_notice_id"] == "1-2013"


def test_a_dry_run_through_run_emits_nothing():
    graph = _Graph([_junk(0)], pairs=[("junk-0", "real-0", "m")])
    out = io.StringIO()
    emit = AsyncMock()
    with patch.object(rj, "get_driver", AsyncMock(return_value=_Driver(graph))), \
         patch.object(rj, "close_driver", AsyncMock()), \
         patch.object(rj.eventlog, "emit_retract_same_as_many", emit):
        asyncio.run(rj.run(rj.Options(dry_run=True, country=None), out))
    assert emit.await_count == 0 and not graph.writes
    assert json.loads(out.getvalue())["dry_run"] is True


def test_cli_defaults_and_report_file(tmp_path):
    report = tmp_path / "c1.json"
    with patch.object(rj, "run", AsyncMock(return_value=True)) as run:
        rj.main(["--report", str(report)])
    opts, out = run.await_args.args
    assert opts == rj.Options()
    assert (opts.dry_run, opts.batch, opts.reviewer, opts.country) == (
        False, 100, rj.DEFAULT_REVIEWER, "ITA")
    assert opts.patterns == rj.DEFAULT_PATTERNS
    assert out.name == str(report)
    assert report.exists()


def test_cli_flags_and_a_short_run_exits_nonzero(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("(?i)^lotto n\\.\n", encoding="utf-8")
    with patch.object(rj, "run", AsyncMock(return_value=False)) as run, \
         pytest.raises(SystemExit) as exc:
        rj.main(["--dry-run", "--batch", "7", "--pattern-file", str(f), "--reviewer", "me",
                 "--country", "PRT"])
    assert exc.value.code == 1
    opts, out = run.await_args.args
    assert (opts.dry_run, opts.batch, opts.reviewer, opts.country) == (True, 7, "me", "PRT")
    assert opts.patterns == (r"(?i)^lotto n\.",)
    assert out is sys.stdout


def test_cli_any_country_lifts_the_filter():
    with patch.object(rj, "run", AsyncMock(return_value=True)) as run:
        rj.main(["--any-country"])
    assert run.await_args.args[0].country is None
