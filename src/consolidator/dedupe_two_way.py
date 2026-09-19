"""Fold equivalence pairs recorded in BOTH directions into one edge.

Until fontem-consolidator#245 the proposal and assertion writes MERGEd a
directed edge, so a pair evaluated from both ends got two: A's run wrote
(A)-[:SAME_AS_CANDIDATE]->(B), B's run wrote (B)-[:SAME_AS_CANDIDATE]->(A).
The review queue showed those pairs twice, and a few were asserted twice
(two :SAME_AS edges). #245 made every write direction-agnostic, so no new
ones appear; this command folds the ones already there.

Each two-way :SAME_AS_CANDIDATE pair becomes one edge, merged the way the
consolidator itself combines two proposals for one pair:

  * the edge that survives is the settled one (approved beats pending);
    between two approved, the first assertion; between two pending, the
    older proposal;
  * detection_rules / _confidences / _dates are unioned per rule, and a
    rule on both keeps its latest firing -- a re-fire replaces, as in
    actions._propose_candidate;
  * a pending edge's summary (confidence, method, detected_at) is
    recomputed from the strongest rule, as _propose_candidate does; an
    approved edge keeps the fields its assertion wrote;
  * conflict is OR-ed, and the first recorded explanation is kept.

Two-way :SAME_AS pairs keep the edge in canonical direction (start key <
end key, the order AssertSameAs events now use) and drop the other.

Safe to run while the sweeper runs: each batch is written only where both
edges still hold what was read (an optimistic check on detection_dates
and status), and whatever changed underneath is picked up by the next
pass. Idempotent: a second run finds nothing to do.

Usage::

    python -m src.consolidator.dedupe_two_way --dry-run
    python -m src.consolidator.dedupe_two_way
"""
from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from src.config import settings
from src.consolidator.neo4j.client import close_driver, get_driver

_READ_CANDIDATE_PAIRS = """
MATCH (a)-[r1:SAME_AS_CANDIDATE]->(b)-[r2:SAME_AS_CANDIDATE]->(a)
WHERE elementId(r1) < elementId(r2)
RETURN elementId(r1) AS id1, properties(r1) AS p1,
       elementId(r2) AS id2, properties(r2) AS p2
LIMIT $batch
"""

# Written only where both edges still hold what was read: the sweeper may
# re-propose or assert either one between the read and this write.
_WRITE_CANDIDATE_PAIRS = """
UNWIND $rows AS row
MATCH ()-[keep:SAME_AS_CANDIDATE]->() WHERE elementId(keep) = row.keep
MATCH ()-[drop:SAME_AS_CANDIDATE]->() WHERE elementId(drop) = row.drop
WITH keep, drop, row
WHERE coalesce(keep.detection_dates, []) = row.keep_dates AND keep.status = row.keep_status
  AND coalesce(drop.detection_dates, []) = row.drop_dates AND drop.status = row.drop_status
SET keep = row.props
DELETE drop
RETURN count(*) AS folded
"""

_SAME_AS_PAIRS = """
MATCH (a)-[s1:SAME_AS]->(b)-[s2:SAME_AS]->(a)
WHERE elementId(s1) < elementId(s2)
RETURN count(*) AS pairs
"""

_FOLD_SAME_AS = """
MATCH (a)-[s1:SAME_AS]->(b)-[s2:SAME_AS]->(a)
WHERE elementId(s1) < elementId(s2)
WITH s1, s2, coalesce(a.gmr_id, a.authority_id) AS ka, coalesce(b.gmr_id, b.authority_id) AS kb
// s1 runs a -> b, so it is the canonical one when ka < kb.
WITH CASE WHEN ka < kb THEN s1 ELSE s2 END AS keep,
     CASE WHEN ka < kb THEN s2 ELSE s1 END AS drop
SET keep.confidence = CASE WHEN coalesce(drop.confidence, 0) > coalesce(keep.confidence, 0)
                           THEN drop.confidence ELSE keep.confidence END
DELETE drop
RETURN count(*) AS folded
"""


def _entries(props: dict) -> list[tuple[str, float, str]]:
    rules = props.get("detection_rules") or []
    confs = props.get("detection_confidences") or []
    dates = props.get("detection_dates") or []
    return list(zip(rules, confs, dates))


def _first_detected(props: dict) -> str:
    dates = props.get("detection_dates") or []
    return min(dates) if dates else (props.get("detected_at") or "")


def _survivor(p1: dict, p2: dict) -> tuple[int, int]:
    """(index of the edge that stays, index of the one folded into it)."""
    approved = [p.get("status") == "approved" for p in (p1, p2)]
    if approved[0] != approved[1]:
        return (0, 1) if approved[0] else (1, 0)
    if all(approved):
        first = (p1.get("decided_at") or "", p2.get("decided_at") or "")
    else:
        first = (_first_detected(p1), _first_detected(p2))
    return (0, 1) if first[0] <= first[1] else (1, 0)


def merge_pair(p1: dict, p2: dict) -> tuple[int, dict]:
    """Merge two proposals for one pair. Returns (index of the surviving
    edge, its full new property map)."""
    keep_i, drop_i = _survivor(p1, p2)
    keep, drop = (p1, p2)[keep_i], (p1, p2)[drop_i]

    # Per rule, the latest firing wins -- what a re-fire does in
    # _propose_candidate. Ordered by firing date.
    latest: dict[str, tuple[float, str]] = {}
    for rule, conf, date in _entries(keep) + _entries(drop):
        if rule not in latest or date >= latest[rule][1]:
            latest[rule] = (conf, date)
    ordered = sorted(latest.items(), key=lambda kv: kv[1][1])

    merged = dict(keep)
    if ordered:
        merged["detection_rules"] = [r for r, _ in ordered]
        merged["detection_confidences"] = [c for _, (c, _) in ordered]
        merged["detection_dates"] = [d for _, (_, d) in ordered]
    merged["conflict"] = bool(keep.get("conflict")) or bool(drop.get("conflict"))
    for field in ("conflict_property", "conflict_left", "conflict_right"):
        if merged.get(field) is None and drop.get(field) is not None:
            merged[field] = drop[field]

    if merged.get("status") != "approved" and ordered:
        # Pending: the summary is the strongest rule, first one on a tie,
        # exactly as _propose_candidate's reduce() picks it.
        top = max(range(len(ordered)), key=lambda i: (ordered[i][1][0], -i))
        rule, (conf, date) = ordered[top]
        merged.update(confidence=conf, method=rule, detected_at=date)
    return keep_i, merged


def _plan(records) -> tuple[list[dict], dict]:
    """Merge every pair read; return the write rows and what they amount to."""
    stats = {"pairs": 0, "kept_approved": 0, "both_approved": 0}
    rows = []
    for rec in records:
        p1, p2 = dict(rec["p1"]), dict(rec["p2"])
        keep_i, props = merge_pair(p1, p2)
        keep_id, drop_id = (rec["id1"], rec["id2"]) if keep_i == 0 else (rec["id2"], rec["id1"])
        keep_p, drop_p = (p1, p2) if keep_i == 0 else (p2, p1)
        stats["pairs"] += 1
        stats["kept_approved"] += props.get("status") == "approved"
        stats["both_approved"] += p1.get("status") == p2.get("status") == "approved"
        rows.append({
            "keep": keep_id, "drop": drop_id, "props": props,
            "keep_dates": keep_p.get("detection_dates") or [], "keep_status": keep_p.get("status"),
            "drop_dates": drop_p.get("detection_dates") or [], "drop_status": drop_p.get("status"),
        })
    return rows, stats


async def _read(driver, database: str, batch: int | None):
    query = _READ_CANDIDATE_PAIRS if batch else _READ_CANDIDATE_PAIRS.replace("LIMIT $batch", "")
    async with driver.session(database=database) as session:
        result = await session.run(query, batch=batch)
        return [rec async for rec in result]


async def fold_candidates(driver, database: str, *, batch: int, dry_run: bool) -> dict:
    """Fold every two-way :SAME_AS_CANDIDATE pair. A dry run reads them all
    once and reports; a real run folds batch by batch until a pass finds
    nothing left it can fold (a pair the sweeper changed mid-batch is read
    again on the next pass)."""
    if dry_run:
        rows, stats = _plan(await _read(driver, database, None))
        if rows:
            logger.info("dry run, first pair: keep {keep}, drop {drop}, props {props}", **rows[0])
        return {**stats, "folded": 0, "passes": 1}
    total = {"pairs": 0, "kept_approved": 0, "both_approved": 0, "folded": 0, "passes": 0}
    while True:
        total["passes"] += 1
        rows, stats = _plan(await _read(driver, database, batch))
        if not rows:
            break
        async with driver.session(database=database) as session:
            result = await session.run(_WRITE_CANDIDATE_PAIRS, rows=rows)
            rec = await result.single()
        folded = int(rec["folded"]) if rec else 0
        for k in ("kept_approved", "both_approved"):
            total[k] += stats[k]
        total["pairs"] += stats["pairs"]
        total["folded"] += folded
        logger.info("pass {passes}: {folded} folded so far", **total)
        if folded == 0:
            # Everything left changed under us twice running; a re-run
            # picks it up. Never spin.
            break
    return total


async def fold_same_as(driver, database: str, *, dry_run: bool) -> int:
    async with driver.session(database=database) as session:
        result = await session.run(_SAME_AS_PAIRS if dry_run else _FOLD_SAME_AS)
        rec = await result.single()
    return int(rec[0]) if rec else 0


async def run(*, batch: int, dry_run: bool) -> None:
    driver = await get_driver()
    try:
        stats = await fold_candidates(driver, settings.neo4j_database, batch=batch, dry_run=dry_run)
        same_as = await fold_same_as(driver, settings.neo4j_database, dry_run=dry_run)
        logger.info(
            "{mode}: two-way :SAME_AS_CANDIDATE pairs {pairs} (folded {folded}; "
            "{kept_approved} kept approved, {both_approved} approved both ways); "
            "two-way :SAME_AS pairs {same_as}",
            mode="DRY RUN" if dry_run else "done", same_as=same_as, **stats,
        )
    finally:
        await close_driver()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    parser.add_argument("--batch", type=int, default=1000, help="pairs per write transaction")
    args = parser.parse_args(argv)
    asyncio.run(run(batch=args.batch, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
