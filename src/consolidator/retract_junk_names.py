"""Retract the equivalences junk-named companies were merged into (C1).

Some TED notices put notice text where the supplier's name belongs --
"Gara aggiudicata come da determina n. 543 del 2013 pubblicata sul sito
www..." -- and each became a :Company. fuzzy_name_same_country then
auto-asserted them equal to one another at >= 0.97 (measured
2026-09-21: 6,683 asserted :SAME_AS edges, 534 nodes with degree > 50,
1,307 companies on the Italian patterns), so every view that follows
the identity class merges hundreds of unrelated awards.

For every :Company whose name matches a junk pattern this command
withdraws each equivalence it takes part in, the way
POST /same-as/{a}/{b}/correct does for one pair:

  1. RetractSameAs is emitted for the pair, one transaction per batch
     of nodes (eventlog.emit_retract_same_as_many). The neo4j sink
     deletes the :SAME_AS edge, the virtuoso sink the owl:sameAs.
  2. Only once the batch has landed is the Neo4j side recorded, through
     the endpoint's own code path (actions.record_correction): the
     settled :SAME_AS_CANDIDATE goes, :NOT_SAME_AS is MERGEd with
     reviewer / reason / retracted_method, a DecisionLog row is written.
     Emitting first means a batch the event store dropped is retried on
     the next run instead of looking settled forever -- the engine's
     flush marks pairs the same way, after the events land.

It also writes a JSON report (--report, default stdout) with, per node,
the raw name, the patterns it matched, the contracts attached to it
(AWARDED_TO: ted_notice_id, contract_key, title) and the pairs
retracted. That report is the input to the next step.

What it deliberately does NOT do: delete the nodes. Removing the
entities and keeping their text on the contracts (C2's shape:
supplier_name_raw on the Contract, the supplier reference dropped) is a
separate, assertion-gated step, and it must not run until this report
has been reviewed and the sinks have consumed the retractions.

Idempotent: a pair already carrying :NOT_SAME_AS is skipped by the read
query, so a re-run only touches what the previous run did not finish
(a batch whose emit failed). Safe to re-run while the sinks are still
catching up: the :SAME_AS edge they have not deleted yet sits behind a
:NOT_SAME_AS and is not re-emitted.

Usage::

    python -m src.consolidator.retract_junk_names --dry-run
    python -m src.consolidator.retract_junk_names --report /tmp/c1.json
    python -m src.consolidator.retract_junk_names --pattern-file p.txt --batch 200

--dry-run only scans: the report has the count per pattern, the
:SAME_AS and AWARDED_TO degrees over the matched nodes, and the first
ten examples; nothing is written or emitted.
--country (default ITA) limits the scan to companies of one country;
--any-country lifts that. The defaults are the Italian family, and
unscoped they reach real names elsewhere: on shared, `vedi ` matched
"... UGYVEDI IRODA" (a Hungarian law office) and "... MEDVEDI PLUS"
(a Czech fund), and `www.` a registered UK name. Word boundaries in the
patterns take care of the first two; the country filter is the belt.
--pattern-file replaces the default patterns, one regex per line (blank
lines and # comments ignored). Patterns are Python regexes and are also
combined into one regex for Neo4j's =~ (Java), so keep to the syntax
both share; the Python side decides, the Cypher side only pre-filters.

A matched name is still spared when it carries a legal-form token
(S.r.l., SRL, S.p.A., Ltd, GmbH, ...) or the node holds a vat or lei:
"GRUPPO VEDI S.R.L." and "WWW.ROBINSONPETSHOP.IT SRL" are companies,
"VEDI ATTI DI AGGIUDICAZIONE" is not, and retracting a real company's
equivalences would block them for good. The plan's own negative signal
(C2, layer 2). Spared nodes are counted and listed in the report.

Run it as its own Job, not by exec'ing into the sweeper or API pod (see
dedupe_two_way for why). A run that stops short exits 1.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TextIO

from loguru import logger

from src.config import settings
from src.consolidator import eventlog
from src.consolidator.actions import Correction, entity_iri, record_correction
from src.consolidator.neo4j.client import close_driver, get_driver

#: The Italian family from data-backlog Part 5 (C1/C2) plus the two
#: language-agnostic hard signals. Searched, not anchored, unless the
#: pattern anchors itself. The phrases are word-bounded (\b means the
#: same in Java and Python): a bare `vedi ` reached "UGYVEDI IRODA" and
#: "MEDVEDI PLUS" on shared, both real names.
DEFAULT_PATTERNS: tuple[str, ...] = (
    r"(?i)^gara aggiudicata",
    r"(?i)\bcome da determina\b",
    r"(?i)\bdetermina n\.",
    r"(?i)\bpubblicat[ao] sul sito\b",
    r"(?i)\bsi veda\b",
    r"(?i)\bvedi\b",
    r"(?i)\baggiudicat[ao] con\b",
    r"(?i)www\.",
    r"(?i)http",
)

#: A name carrying one of these is a company however sentence-like the
#: rest of it reads. Token-bounded (not preceded by a word char or a
#: dot, not followed by one), dots optional, Italian forms first; only
#: forms of three letters or more, or dotted, because a bare "AD"/"AS"
#: is a word in Italian notice text.
LEGAL_FORM_RE = re.compile(
    r"(?i)(?<![\w.])(?:"
    r"s\.?r\.?l\.?s?|s\.?p\.?a\.?|s\.?a\.?s\.?|s\.?n\.?c\.?|s\.?c\.?a\.?r\.?l\.?|"
    r"s\.?c\.?p\.?a\.?|s\.?c\.?s\.?|s\.?c\.? a r\.?l\.?|"
    r"responsabilit[àa]'? limitata|per azioni|in accomandita|in nome collettivo|"
    r"soc\.? ?coop\.?|societ[àa]'? cooperativa|cooperativa|consorzio|"
    r"ltd\.?|limited|plc|llp|llc|inc\.?|gmbh|a\.g\.?|kgaa|e\.v\.?|"
    r"s\.a\.?|s\.l\.?|lda|s\.?a\.?r\.?l\.?|s\.?à ?r\.?l\.?|sasu|eurl|"
    r"b\.v\.?|n\.v\.?|oyj?|a\.b\.?|a/s|aps|asa|a\.s\.?|"
    r"sp\.? ?z ?o\.? ?o\.?|s\.?r\.?o\.?|kft|zrt|nyrt|d\.?o\.?o\.?|d\.d\.?|"
    r"o\.?o\.?d\.?|e\.?o\.?o\.?d\.?|a\.d\.?|uab|sia|oü|ehf"
    r")(?!\w)"
)

#: The defaults are Italian notice text, so the default scan is Italian
#: companies (Company.country is ISO-3).
DEFAULT_COUNTRY = "ITA"
REASON = "junk name (data-backlog C1): notice text where the supplier's name belongs"
DEFAULT_REVIEWER = "retract_junk_names"
EXAMPLES = 10

#: Reads the :SAME_AS degree; the edge itself is only ever written by the
#: sink (test_merge_no_self_loops guards that boundary by text, hence
#: the bound variable).
_SCAN_MATCH = "MATCH (c:Company) WHERE c.name =~ $regex"
_SCAN_RETURN = """
RETURN c.gmr_id AS gmr_id, c.name AS name,
       (c.vat IS NOT NULL OR c.lei IS NOT NULL) AS has_hard_id,
       size([(c)-[s:SAME_AS]-() | 1]) AS same_as_degree,
       size([(c)-[:AWARDED_TO]-() | 1]) AS contract_degree
"""


@dataclass(frozen=True)
class Options:
    """One run's settings, as the CLI collects them."""

    dry_run: bool = False
    batch: int = 100
    reviewer: str = DEFAULT_REVIEWER
    patterns: tuple[str, ...] = DEFAULT_PATTERNS
    country: str | None = DEFAULT_COUNTRY

#: Every equivalence a junk node takes part in that has not been
#: corrected: an asserted :SAME_AS edge, or an approved candidate (an
#: approval is an emitted AssertSameAs whether or not the sink has
#: projected the edge yet). :NOT_SAME_AS is what a previous run, or an
#: operator, left behind -- the idempotency guard.
_PAIRS = """
UNWIND $ids AS id
MATCH (c:Company {gmr_id: id})-[e:SAME_AS|SAME_AS_CANDIDATE]-(o:Company)
WHERE (type(e) = 'SAME_AS' OR e.status = 'approved')
  AND o.gmr_id <> c.gmr_id
  AND NOT EXISTS { (c)-[:NOT_SAME_AS]-(o) }
WITH c, o, collect(coalesce(e.method, e.rule)) AS methods
RETURN c.gmr_id AS gmr_id, o.gmr_id AS other_id, o.name AS other_name,
       methods[0] AS method
"""

_CONTRACTS = """
UNWIND $ids AS id
MATCH (c:Company {gmr_id: id})-[:AWARDED_TO]-(k:Contract)
RETURN id AS gmr_id, k.ted_notice_id AS ted_notice_id,
       k.contract_key AS contract_key, k.title AS title
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── patterns ──────────────────────────────────────────────────────────


def load_patterns(path: str | None) -> list[str]:
    """The defaults, or one regex per non-blank, non-comment line of a file."""
    if path is None:
        return list(DEFAULT_PATTERNS)
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def matched_patterns(name: str | None, compiled: list[re.Pattern[str]]) -> list[str]:
    """The patterns that hit this name, in the order given. This is the
    decision; the Cypher pre-filter only narrows the scan."""
    if not name:
        return []
    return [rx.pattern for rx in compiled if rx.search(name)]


def skip_reason(name: str | None, has_hard_id: bool) -> str | None:
    """Why a matched name is spared: it holds a hard identifier, or it
    carries a legal form. None when it is junk to act on."""
    if has_hard_id:
        return "hard_id"
    if name and LEGAL_FORM_RE.search(name):
        return "legal_form"
    return None


def cypher_regex(patterns: list[str]) -> str:
    """One Java regex for the set, for Neo4j's =~ (a full match): each
    pattern is wrapped in .* unless it anchors itself, inline (?i) is
    hoisted, (?s) so a newline inside a name does not hide it. Case-
    insensitive across the board, so it can over-match a pattern
    without (?i); matched_patterns then drops those rows."""
    parts = []
    for p in patterns:
        body = p[4:] if p.startswith("(?i)") else p
        parts.append(f"(?:{body}).*" if body.startswith("^") else f".*(?:{body}).*")
    return "(?is)(?:" + "|".join(parts) + ")"


# ── scan ──────────────────────────────────────────────────────────────


def scan_query(country: str | None) -> str:
    """The label scan, narrowed to one country when asked."""
    where = " AND c.country = $country" if country else ""
    return _SCAN_MATCH + where + _SCAN_RETURN


async def scan(
    driver, database: str, patterns: list[str], country: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """(junk nodes to act on, matched nodes spared) -- each with the
    patterns that hit and its :SAME_AS / AWARDED_TO degrees; a spared
    node also says why (`skipped`). Streamed off the driver and kept:
    1,307 nodes on prod for the default patterns, a few MB at worst."""
    compiled = compile_patterns(patterns)
    nodes: list[dict] = []
    skipped: list[dict] = []
    async with driver.session(database=database) as session:
        result = await session.run(
            scan_query(country), regex=cypher_regex(patterns), country=country,
        )
        async for rec in result:
            hit = matched_patterns(rec["name"], compiled)
            if not hit:
                continue
            node = {
                "gmr_id": rec["gmr_id"], "name": rec["name"], "patterns": hit,
                "same_as_degree": int(rec["same_as_degree"] or 0),
                "contract_degree": int(rec["contract_degree"] or 0),
            }
            reason = skip_reason(rec["name"], bool(rec.get("has_hard_id")))
            if reason:
                skipped.append({**node, "skipped": reason})
            else:
                nodes.append(node)
    return nodes, skipped


def _by_pattern(nodes: list[dict], patterns: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(patterns, 0)
    for n in nodes:
        for p in n["patterns"]:
            counts[p] += 1
    return counts


def dry_run_report(
    nodes: list[dict], patterns: list[str], country: str | None = None,
    skipped: list[dict] | None = None,
) -> dict:
    """What a run would touch: counts per pattern, the degrees summed
    over the junk nodes, the first EXAMPLES of them verbatim -- and what
    matched but was spared, so the operator can see the guard at work."""
    skipped = skipped or []
    return {
        "generated_at": _now(),
        "dry_run": True,
        "patterns": list(patterns),
        "country": country,
        "counts": {
            "nodes": len(nodes),
            "by_pattern": _by_pattern(nodes, patterns),
            "same_as_edges": sum(n["same_as_degree"] for n in nodes),
            "contracts": sum(n["contract_degree"] for n in nodes),
            "skipped": {
                reason: sum(n["skipped"] == reason for n in skipped)
                for reason in ("legal_form", "hard_id")
            },
        },
        "examples": nodes[:EXAMPLES],
        "skipped_examples": skipped[:EXAMPLES],
    }


# ── retract ───────────────────────────────────────────────────────────


async def _read_pairs(session, ids: list[str]) -> list[dict]:
    """The uncorrected pairs of these nodes, each once: when both ends
    are junk nodes in the same batch the read returns it from both
    sides, and one retraction settles it."""
    result = await session.run(_PAIRS, ids=ids)
    pairs: list[dict] = []
    seen: set[frozenset[str]] = set()
    async for rec in result:
        key = frozenset((rec["gmr_id"], rec["other_id"]))
        if key not in seen:
            seen.add(key)
            pairs.append(dict(rec))
    return pairs


async def _read_contracts(session, ids: list[str]) -> dict[str, list[dict]]:
    result = await session.run(_CONTRACTS, ids=ids)
    out: dict[str, list[dict]] = {}
    async for rec in result:
        out.setdefault(rec["gmr_id"], []).append({
            k: rec[k] for k in ("ted_notice_id", "contract_key", "title")
        })
    return out


def retract_rows(pairs: list[dict], reviewer: str) -> list[dict]:
    """The rows emit_retract_same_as_many takes. a_iri is the junk node,
    b_iri its partner -- the direction record_correction writes the
    :NOT_SAME_AS edge in, so the sink's MERGE lands on the same edge."""
    return [{
        "a_iri": entity_iri("Company", p["gmr_id"]),
        "b_iri": entity_iri("Company", p["other_id"]),
        "reason": REASON,
        "reviewer": reviewer,
        "retracted_method": p.get("method"),
        "domain": "company",
    } for p in pairs]


async def _record(session, pairs: list[dict], reviewer: str) -> None:
    for p in pairs:
        await record_correction(session, Correction(
            label="Company", from_id=p["gmr_id"], to_id=p["other_id"],
            reviewer=reviewer, reason=REASON, retracted_method=p.get("method"),
        ))


async def _retract_batch(session, ids: list[str], reviewer: str) -> tuple[list[dict], bool]:
    """Emit the batch's retractions, then record them. Returns the pairs
    and whether they landed; on a short batch nothing is recorded, so
    the next run finds the same pairs again."""
    pairs = await _read_pairs(session, ids)
    if not pairs:
        return pairs, True
    sent = await eventlog.emit_retract_same_as_many(retract_rows(pairs, reviewer))
    if sent != len(pairs):
        logger.error(
            "retract: {sent}/{n} RetractSameAs landed for this batch; "
            "nothing recorded, stopping", sent=sent, n=len(pairs),
        )
        return pairs, False
    await _record(session, pairs, reviewer)
    return pairs, True


def _partners(node_id: str, pairs: list[dict]) -> list[dict]:
    out = []
    for p in pairs:
        if node_id not in (p["gmr_id"], p["other_id"]):
            continue
        mine = p["gmr_id"] == node_id
        out.append({
            "other_id": p["other_id"] if mine else p["gmr_id"],
            "other_name": p.get("other_name") if mine else None,
            "method": p.get("method"),
        })
    return out


async def _process_batch(
    session, chunk: list[dict], reviewer: str,
) -> tuple[list[dict], int, bool]:
    """One batch: contracts, pairs, emit, record. Returns the report
    entries, the pairs retracted, and whether the emit landed."""
    ids = [n["gmr_id"] for n in chunk]
    contracts = await _read_contracts(session, ids)
    pairs, landed = await _retract_batch(session, ids, reviewer)
    entries = [{
        **n,
        "contracts": contracts.get(n["gmr_id"], []),
        "same_as": _partners(n["gmr_id"], pairs),
        "retracted": landed,
    } for n in chunk]
    return entries, len(pairs) if landed else 0, landed


async def retract(
    driver, database: str, nodes: list[dict], *, batch: int, reviewer: str,
) -> tuple[list[dict], int, bool]:
    """The run over the scanned nodes, batch by batch. Stops at the first
    batch whose events did not land: (entries, pairs retracted, complete)."""
    entries: list[dict] = []
    retracted, complete = 0, True
    async with driver.session(database=database) as session:
        for start in range(0, len(nodes), batch):
            got, n, complete = await _process_batch(
                session, nodes[start:start + batch], reviewer,
            )
            entries.extend(got)
            retracted += n
            logger.info(
                "retract: {done}/{total} nodes, {retracted} pairs retracted",
                done=len(entries), total=len(nodes), retracted=retracted,
            )
            if not complete:
                break
    return entries, retracted, complete


def run_report(base: dict, entries: list[dict], retracted: int, complete: bool) -> dict:
    """The dry-run report plus what the run did: per-node entries with
    their contracts and retracted pairs, and whether it finished."""
    return {
        **base,
        "dry_run": False,
        "counts": {
            **base["counts"],
            "pairs": sum(len(e["same_as"]) for e in entries),
            "retracted": retracted,
        },
        "complete": complete,
        "examples": None,
        "nodes": entries,
    }


async def run(opts: Options, out: TextIO) -> bool:
    """Scan, optionally retract, write the report. True when complete."""
    driver = await get_driver()
    patterns = list(opts.patterns)
    try:
        nodes, skipped = await scan(driver, settings.neo4j_database, patterns, opts.country)
        report = dry_run_report(nodes, patterns, opts.country, skipped)
        logger.info(
            "{mode}: {nodes} junk-named companies in {country}, {edges} :SAME_AS "
            "edges, {contracts} contracts; per pattern {by_pattern}; spared {skipped}",
            mode="DRY RUN" if opts.dry_run else "scan", nodes=len(nodes),
            country=opts.country or "any country",
            edges=report["counts"]["same_as_edges"],
            contracts=report["counts"]["contracts"],
            by_pattern=report["counts"]["by_pattern"],
            skipped=report["counts"]["skipped"],
        )
        if not opts.dry_run:
            entries, retracted, complete = await retract(
                driver, settings.neo4j_database, nodes,
                batch=opts.batch, reviewer=opts.reviewer,
            )
            report = run_report(report, entries, retracted, complete)
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write("\n")
        return bool(report.get("complete", True))
    finally:
        await close_driver()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="scan and report; retract nothing")
    parser.add_argument("--batch", type=int, default=100,
                        help="junk nodes per RetractSameAs transaction")
    parser.add_argument("--pattern-file", help="one regex per line, replacing the defaults")
    parser.add_argument("--report", help="write the JSON report here (default: stdout)")
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER,
                        help="recorded on :NOT_SAME_AS and DecisionLog")
    parser.add_argument("--country", default=DEFAULT_COUNTRY,
                        help="ISO-3; scan only companies of this country")
    parser.add_argument("--any-country", action="store_true",
                        help="lift the country filter (the dry run shows the sweep)")
    args = parser.parse_args(argv)
    opts = Options(
        dry_run=args.dry_run, batch=args.batch, reviewer=args.reviewer,
        patterns=tuple(load_patterns(args.pattern_file)),
        country=None if args.any_country else args.country,
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as out:
            complete = asyncio.run(run(opts, out))
    else:
        complete = asyncio.run(run(opts, sys.stdout))
    if not complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
