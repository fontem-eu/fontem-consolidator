"""Re-assert the equivalences Neo4j is missing, so :SAME_AS exists.

The consolidator has 39,193 approved :SAME_AS_CANDIDATE edges and
Virtuoso holds the matching owl:sameAs, but Neo4j holds no :SAME_AS at
all: the neo4j sink mapped AssertSameAs to None for the period when
nothing in Neo4j followed the edge, so every assertion in that window
was read off the log and dropped on the floor.

The read path follows it now (fontem-api#434), so the edges have to be
there. Two ways to get them:

Replaying the sink over the AssertSameAs events would be the purest --
they are all still in the log. But they span seq 4,663,845 to 7,300,805
with 2.69 MILLION events in between, and the consumer reads
``WHERE seq > offset`` with no type filter, so "replay the assertions"
means replaying everything else too.

So: re-emit. One AssertSameAs per currently-approved candidate, at the
head of the log. Each restates a fact that is already true, the sink's
MERGE is idempotent, and afterwards the edge is derivable from the log
the same way every other fact is.

What decides the set
--------------------
The Neo4j candidate graph, not the old events. 47,640 AssertSameAs were
emitted and 7,872 later retracted, so replaying history would assert
pairs an operator has since separated and leave the graph contradicting
its own :NOT_SAME_AS edges (DQ assertion refs.sameas_not_contradicted,
BLOCK). Reading the current approved candidates instead means the
backfill states today's truth. Pairs carrying a :NOT_SAME_AS are
excluded explicitly rather than trusted to be absent.

Usage::

    python -m src.consolidator.backfill_same_as            # report
    python -m src.consolidator.backfill_same_as --apply    # emit
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from src.config import settings
from src.consolidator import eventlog
from src.consolidator.actions import entity_iri
from src.consolidator.neo4j.client import close_driver, get_driver

#: Approved equivalences that Neo4j can still address, with the pair's
#: own correction edge excluded. Ordered so a re-run emits the same
#: sequence, which makes a partial run resumable by inspection.
_FIND = """
MATCH (a)-[r:SAME_AS_CANDIDATE {status: 'approved'}]->(b)
WHERE NOT EXISTS { (a)-[:NOT_SAME_AS]-(b) }
  AND labels(a)[0] = labels(b)[0]
  AND elementId(a) <> elementId(b)
WITH a, b, r, labels(a)[0] AS label,
     coalesce(a.gmr_id, a.authority_id, a.person_id) AS ak,
     coalesce(b.gmr_id, b.authority_id, b.person_id) AS bk
WHERE ak IS NOT NULL AND bk IS NOT NULL AND ak <> bk
RETURN label, ak, bk,
       coalesce(r.confidence, 1.0) AS confidence,
       coalesce(r.method, 'backfill') AS method,
       r.rule AS rule
ORDER BY label, ak, bk
"""


async def find_approved() -> list[dict]:
    """Every approved pair the backfill should assert."""
    driver = await get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(_FIND)
        raw = [dict(r) async for r in result]
    rows = []
    for r in raw:
        rows.append({
            # Minted by the shared helper, not concatenated in Cypher:
            # the sink parses these IRIs straight back into (label, key)
            # to find the nodes, so a scheme that drifts from the one
            # actions.py mints is an edge that silently never appears.
            "a_iri": entity_iri(r["label"], r["ak"]),
            "b_iri": entity_iri(r["label"], r["bk"]),
            "confidence": r["confidence"],
            "method": r["method"],
            "rule": r["rule"],
            # domain is the event envelope's, not the payload's; the
            # consolidator emits every equivalence under one domain.
            "domain": "consolidation",
        })
    return rows


async def backfill(*, apply: bool, batch: int = 500) -> int:
    """Emit an AssertSameAs per approved pair. Returns events written."""
    rows = await find_approved()
    logger.info("backfill: {n} approved equivalences to assert", n=len(rows))
    if not apply:
        logger.info("dry run: would emit {n} AssertSameAs events", n=len(rows))
        return 0
    sent = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        # emit_assert_same_as_many swallows failures by design -- a
        # flaky event store must not abort a consolidation run. Here
        # that would silently under-fill the graph, so count what
        # actually landed and say so.
        got = await eventlog.emit_assert_same_as_many(chunk)
        sent += got
        if got != len(chunk):
            logger.warning(
                "backfill: emitted {got}/{want} in chunk at offset {off}",
                got=got, want=len(chunk), off=start,
            )
        logger.info("backfill: emitted {sent}/{total}", sent=sent,
                    total=len(rows))
    return sent


def main(argv=None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Emit the events. Without this the script only reports.",
    )
    parser.add_argument("--batch", type=int, default=500)
    args = parser.parse_args(argv)
    async def _run() -> int:
        try:
            return await backfill(apply=args.apply, batch=args.batch)
        finally:
            await close_driver()

    sent = asyncio.run(_run())
    logger.info("backfill: done, {n} events emitted", n=sent)


if __name__ == "__main__":
    main()
