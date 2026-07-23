"""consolidator-sweeper — continuous re-consolidation of the entity graph.

Run as ``python -m src.consolidator.sweeper``.

Why this exists
---------------
The ``consolidator-trigger`` only consolidates an entity when *its own*
event is dispatched. While the trigger sat scaled-to-0 for months every
entity upserted in that window was never evaluated, so byte-identical
duplicates (e.g. two identical "Mészáros és Mészáros … Zrt." Company
nodes in the same country) were never flagged. A merge also rewrites a
node's neighbourhood, so duplicate detection only *converges* after
repeated passes. This service gives every Company / Authority a
periodic, event-independent re-evaluation and rotates over the whole
graph forever, oldest-first.

How it rotates
--------------
One async task per label. Each task pages the stalest entities
(oldest — or never-stamped — ``last_consolidated_at`` first), runs the
match-only consolidation pipeline on each id, then stamps
``n.last_consolidated_at = datetime()``. That stamp IS the rotation
cursor: it lives on the node, so the sweep resumes exactly where it
left off across pod restarts, and once every node carries a fresh
stamp the oldest-first order naturally cycles back to the start.

Match-only, no GDS
------------------
``mode="match_only"`` skips the enrichment/translation rules (they need
the linguistics/Mistral backend and are orthogonal to dedup), and
``exclude_rule_prefix="gds_"`` skips the GDS rules (they reproject the
whole subgraph per call — far too expensive to run per-entity in a
tight loop; they run as separate batch jobs).

Config (env)
------------
  SWEEP_LABELS                 default "Company,Authority"
  SWEEP_PAGE_SIZE              default 200
  SWEEP_COMPANY_RATE_PER_SEC   default 6   (~3.6M companies / ~1 week)
  SWEEP_AUTHORITY_RATE_PER_SEC default 2   (~165k authorities / ~1 day)
  SWEEP_EMPTY_BACKOFF_SEC      default 30  (sleep when a page is empty)
  METRICS_PORT                 default 9100
  CONSOLIDATOR_NEO4J_*         Neo4j creds (via src.config.settings)
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass, field

from loguru import logger
from neo4j import AsyncDriver
from prometheus_client import Counter, Gauge, start_http_server

from src.config import settings
from src.consolidator import engine
from src.consolidator.entities import id_key_for
from src.consolidator.neo4j import migrations
from src.consolidator.neo4j.client import close_driver, get_driver
from src.consolidator.rules.loader import load_all as load_rules

# Per-label defaults. ~3.6M companies at 6/s rotate in ~7 days; ~165k
# authorities at 2/s rotate in ~23h. Any label not listed falls back to
# the authority rate (the conservative choice for a smaller-but-unknown
# population).
_DEFAULT_RATES: dict[str, float] = {"Company": 6.0, "Authority": 2.0}
_FALLBACK_RATE = 2.0

# Metrics — registered on the default registry so start_http_server
# exposes them alongside the engine's gmr_consolidator_rule_fires_total.
SWEEP_ENTITIES = Counter(
    "consolidator_sweep_entities_total",
    "Entities re-consolidated by the sweeper, by label and coarse outcome",
    ["label", "outcome"],
)
ROTATION_LAG = Gauge(
    "consolidator_sweep_rotation_lag_seconds",
    "Age of the stalest entity for this label (now - oldest last_consolidated_at)",
    ["label"],
)
SWEEP_RATE = Gauge(
    "consolidator_sweep_rate",
    "Configured re-consolidation rate (entities/sec) for this label",
    ["label"],
)


@dataclass
class SweeperConfig:
    labels: list[str]
    page_size: int
    rates: dict[str, float] = field(default_factory=dict)
    empty_backoff_s: float = 30.0
    metrics_port: int = 9100

    @classmethod
    def from_env(cls) -> "SweeperConfig":
        labels = [
            part.strip()
            for part in os.environ.get("SWEEP_LABELS", "Company,Authority").split(",")
            if part.strip()
        ]
        rates: dict[str, float] = {}
        for label in labels:
            env_name = f"SWEEP_{label.upper()}_RATE_PER_SEC"
            default = _DEFAULT_RATES.get(label, _FALLBACK_RATE)
            rates[label] = float(os.environ.get(env_name, str(default)))
        return cls(
            labels=labels,
            page_size=int(os.environ.get("SWEEP_PAGE_SIZE", "200")),
            rates=rates,
            empty_backoff_s=float(os.environ.get("SWEEP_EMPTY_BACKOFF_SEC", "30")),
            metrics_port=int(os.environ.get("METRICS_PORT", "9100")),
        )


class _Pacer:
    """Steady-rate limiter. ``wait()`` sleeps just enough to keep at most
    ``rate`` acquisitions per second, smoothing them evenly rather than
    bursting a page then idling. Clock + sleep are injectable so the
    pacing maths is unit-testable without real time."""

    def __init__(self, rate_per_sec: float, *, sleep=asyncio.sleep, clock=time.monotonic):
        self._interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._sleep = sleep
        self._clock = clock
        self._next: float | None = None

    async def wait(self) -> None:
        if self._interval <= 0:
            return
        now = self._clock()
        if self._next is None:
            self._next = now
        delay = self._next - now
        if delay > 0:
            await self._sleep(delay)
        # Anchor the next slot off max(now, scheduled) so a slow
        # consolidate() that overran its slot doesn't build up a debt
        # the pacer then tries to "catch up" by never sleeping.
        self._next = max(now, self._next) + self._interval


def _classify(result: engine.ConsolidationResult) -> str:
    """Collapse a run's per-decision outcomes into one coarse label for
    the metric. Priority mirrors engine._summarize:
    merge > link > conflict > flag > noop."""
    outcomes = {d["outcome"] for d in result.decisions}
    if "auto_merge" in outcomes:
        return "merged"
    if "auto_link" in outcomes:
        return "linked"
    if "conflict" in outcomes:
        return "conflict"
    if "flag" in outcomes:
        return "flagged"
    return "noop"


async def _page_stalest(
    driver: AsyncDriver, database: str, label: str, key: str, page_size: int
) -> list[str]:
    query = (
        f"MATCH (n:{label}) WHERE n.name IS NOT NULL "
        f"RETURN n.{key} AS id "
        "ORDER BY coalesce(n.last_consolidated_at, datetime('1970-01-01')) ASC "
        "LIMIT $page"
    )
    async with driver.session(database=database) as session:
        result = await session.run(query, page=page_size)
        return [record["id"] async for record in result if record["id"] is not None]


async def _measure_lag(driver: AsyncDriver, database: str, label: str) -> float:
    """now - oldest last_consolidated_at, in seconds. Computed in Cypher
    so we don't have to marshal Neo4j datetimes into Python."""
    query = (
        f"MATCH (n:{label}) WHERE n.name IS NOT NULL "
        "WITH min(coalesce(n.last_consolidated_at, datetime('1970-01-01'))) AS oldest "
        "RETURN duration.inSeconds(oldest, datetime()).seconds AS lag"
    )
    async with driver.session(database=database) as session:
        result = await session.run(query)
        record = await result.single()
    if record is None or record["lag"] is None:
        return 0.0
    return float(record["lag"])


async def _stamp(
    driver: AsyncDriver, database: str, label: str, key: str, entity_id: str
) -> None:
    query = f"MATCH (n:{label} {{{key}: $id}}) SET n.last_consolidated_at = datetime()"
    async with driver.session(database=database) as session:
        await session.run(query, id=entity_id)


async def _interruptible_sleep(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def sweep_label(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    driver: AsyncDriver,
    database: str,
    label: str,
    *,
    stop_event: asyncio.Event,
    page_size: int,
    rate: float,
    empty_backoff_s: float,
) -> None:
    """Forever: page the stalest entities for ``label`` and re-consolidate
    each, rate-limited, stamping the rotation cursor as we go."""
    key = id_key_for(label)
    pacer = _Pacer(rate)
    SWEEP_RATE.labels(label=label).set(rate)
    logger.info(
        "sweeper[{label}]: starting (key={key}, page={page}, rate={rate}/s)",
        label=label, key=key, page=page_size, rate=rate,
    )
    while not stop_event.is_set():
        try:
            ROTATION_LAG.labels(label=label).set(
                await _measure_lag(driver, database, label)
            )
            ids = await _page_stalest(driver, database, label, key, page_size)
        # A transient Neo4j hiccup on the page/lag queries must not kill
        # the task — back off and retry the page.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "sweeper[{label}]: page query failed ({err}); backing off {s}s",
                label=label, err=repr(exc), s=empty_backoff_s,
            )
            await _interruptible_sleep(stop_event, empty_backoff_s)
            continue

        if not ids:
            logger.info(
                "sweeper[{label}]: no entities to sweep; backing off {s}s",
                label=label, s=empty_backoff_s,
            )
            await _interruptible_sleep(stop_event, empty_backoff_s)
            continue

        for entity_id in ids:
            if stop_event.is_set():
                break
            await pacer.wait()
            outcome = "error"
            try:
                result = await engine.consolidate(
                    driver,
                    database,
                    entity_type=label,
                    entity_id=entity_id,
                    triggered_by="sweeper",
                    exclude_rule_prefix="gds_",
                    mode="match_only",
                )
                outcome = _classify(result)
            # A poison entity must never wedge the rotation: log, count
            # it as an error, and — crucially — still stamp it below so
            # the cursor advances past it on the next page.
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "sweeper[{label}]: consolidate failed for {id} ({err}); continuing",
                    label=label, id=entity_id, err=repr(exc),
                )
            finally:
                try:
                    await _stamp(driver, database, label, key, entity_id)
                # If even the stamp fails the entity stays stale and is
                # retried next rotation — acceptable, just log it.
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "sweeper[{label}]: stamp failed for {id} ({err})",
                        label=label, id=entity_id, err=repr(exc),
                    )
            SWEEP_ENTITIES.labels(label=label, outcome=outcome).inc()


async def run(config: SweeperConfig | None = None) -> None:
    config = config or SweeperConfig.from_env()
    driver = await get_driver()
    # The sweeper is its own process (separate Deployment from the API
    # pod), so it ensures its own indexes — notably the
    # {company,authority}_last_consolidated range indexes the oldest-
    # first page depends on. All statements are IF NOT EXISTS / idempotent.
    await migrations.apply(driver, settings.neo4j_database)
    load_rules()
    start_http_server(config.metrics_port)
    logger.info(
        "sweeper: metrics on :{port}, labels={labels}",
        port=config.metrics_port, labels=config.labels,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    tasks = [
        asyncio.create_task(
            sweep_label(
                driver,
                settings.neo4j_database,
                label,
                stop_event=stop_event,
                page_size=config.page_size,
                rate=config.rates.get(label, _FALLBACK_RATE),
                empty_backoff_s=config.empty_backoff_s,
            ),
            name=f"sweep-{label}",
        )
        for label in config.labels
    ]

    await stop_event.wait()
    logger.info("sweeper: shutdown signal received, draining tasks")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await close_driver()
    logger.info("sweeper: stopped cleanly")


def main() -> None:
    logger.info("sweeper: booting")
    asyncio.run(run())


if __name__ == "__main__":
    main()
