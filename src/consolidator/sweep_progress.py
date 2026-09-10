"""Report how far the full-corpus re-consolidation has got, and when it ends.

The sweeper walks every entity ordered by `last_consolidated_at`, oldest
first, stamping each as it goes. That makes a full pass measurable
without any extra bookkeeping: pick the moment the pass began, and count
how many entities still carry a stamp older than it.

Why a script rather than the existing gauge
-------------------------------------------
`consolidator_sweep_rotation_lag_seconds` reports the age of the STALEST
entity, which answers "is anything being starved" but not "how much is
left". After a reset it reads ~1.789e9 (the 1970 coalesce default) and
stays pinned there until the very last never-consolidated entity is
picked up, so it looks identical at 1% and 99% done.

Rate comes from measurement, not from config. The configured ceiling on
prod is 6/sec for Company; the observed rate on 2026-09-10 was 3.33/sec,
because the limiter is a ceiling and the consolidation work downstream is
what actually paces the sweep. An ETA computed from the config would have
been optimistic by a factor of ~2.

Usage::

    python -m src.consolidator.sweep_progress --since 2026-09-10T22:50:00Z
    python -m src.consolidator.sweep_progress --since ... --sample-seconds 120
"""

from __future__ import annotations

import argparse
import asyncio
import time

from loguru import logger

from src.config import settings
from src.consolidator.neo4j.client import close_driver, get_driver

#: Mirrors the sweeper's own population filter (_page_stalest): an entity
#: with no name is never swept, so counting it as outstanding would mean
#: the pass could never reach 100%.
_SWEEPABLE = "MATCH (n:{label}) WHERE n.name IS NOT NULL"

_COUNTS = _SWEEPABLE + """
RETURN count(n) AS total,
       count(CASE WHEN n.last_consolidated_at >= datetime($since)
                  THEN 1 END) AS done
"""


async def _counts(label: str, since: str) -> tuple[int, int]:
    driver = await get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(_COUNTS.format(label=label), since=since)
        rec = await result.single()
    return int(rec["total"]), int(rec["done"])


async def _measured_rate(label: str, since: str, seconds: int) -> float:
    """Entities/sec, observed over a live window.

    Sampling the graph rather than the Prometheus counter so this works
    against any environment without needing to reach the sweeper's
    metrics port, and so a sweeper restart (which zeroes the counter)
    cannot read as negative progress.
    """
    _, before = await _counts(label, since)
    await asyncio.sleep(seconds)
    _, after = await _counts(label, since)
    return (after - before) / seconds if seconds else 0.0


def _human(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    days, rem = divmod(int(seconds), 86400)
    hours = rem // 3600
    return f"{days}d {hours}h"


async def report(labels: list[str], since: str, sample_seconds: int) -> None:
    """Print coverage and an ETA per label."""
    for label in labels:
        total, done = await _counts(label, since)
        remaining = total - done
        pct = (done / total * 100) if total else 0.0
        logger.info(
            "{label}: {done:,}/{total:,} re-consolidated since {since} "
            "({pct:.2f}%), {remaining:,} remaining",
            label=label, done=done, total=total, since=since,
            pct=pct, remaining=remaining,
        )
        if not remaining:
            logger.info("{label}: full pass COMPLETE", label=label)
            continue
        rate = await _measured_rate(label, since, sample_seconds)
        if rate <= 0:
            logger.warning(
                "{label}: no progress observed in {s}s -- the sweeper may be "
                "stopped, or paced slower than this sample window can see",
                label=label, s=sample_seconds,
            )
            continue
        logger.info(
            "{label}: {rate:.2f}/sec observed -> ETA {eta} "
            "(finishes around {when})",
            label=label, rate=rate, eta=_human(remaining / rate),
            when=time.strftime(
                "%Y-%m-%d %H:%M UTC",
                time.gmtime(time.time() + remaining / rate)),
        )


def main(argv=None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since", required=True,
        help="ISO-8601 instant the pass began, e.g. 2026-09-10T22:50:00Z. "
             "Entities stamped at or after this count as done.",
    )
    parser.add_argument("--labels", default="Company,Authority")
    parser.add_argument(
        "--sample-seconds", type=int, default=120,
        help="Window used to measure the live rate. Too short and the "
             "sweeper's paging makes the rate look lumpy.",
    )
    args = parser.parse_args(argv)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    async def _run() -> None:
        try:
            await report(labels, args.since, args.sample_seconds)
        finally:
            await close_driver()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
