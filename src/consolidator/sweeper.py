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
Only a completed evaluation is stamped (see ``_sweep_one``): a Neo4j
outage or a SIGTERM mid-evaluation leaves the entity stale, so it is
the first one picked up again rather than skipped for a rotation.

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
  SWEEP_CONCURRENCY            default 1   (entities in flight, ALL labels together)
  SWEEP_UNSTAMPED_SCAN_SEC     default 300 (how often to look for never-swept entities)
  METRICS_PORT                 default 9100
  CONSOLIDATOR_NEO4J_*         Neo4j creds (via src.config.settings)
"""
from __future__ import annotations

import asyncio
import os
import random
import signal
import time
from dataclasses import dataclass, field

from loguru import logger
from neo4j import AsyncDriver
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
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
SWEEP_CONCURRENCY = Gauge(
    "consolidator_sweep_concurrency",
    "Configured maximum entities being re-consolidated at once, all labels together",
)


@dataclass
class SweeperConfig:
    labels: list[str]
    page_size: int
    rates: dict[str, float] = field(default_factory=dict)
    empty_backoff_s: float = 30.0
    metrics_port: int = 9100
    # Entities in flight at once across every label. The work is almost
    # all waiting on Neo4j, so one at a time left the sweep at ~3/s with
    # Neo4j mostly idle (prod, 2026-09-18). This is the knob that bounds
    # what the sweep can ask of Neo4j and the event store at any moment;
    # the per-label rates stay as ceilings on top of it.
    concurrency: int = 1
    # How often to look for entities that have never been swept. They
    # are the only ones the last_consolidated_at index cannot find (a
    # range index holds no nulls), so finding them is a full label scan;
    # doing that once per page instead cost two such scans a page.
    unstamped_scan_s: float = 300.0

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
            concurrency=max(1, int(os.environ.get("SWEEP_CONCURRENCY", "1"))),
            unstamped_scan_s=float(os.environ.get("SWEEP_UNSTAMPED_SCAN_SEC", "300")),
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
        # Book the slot BEFORE sleeping. With several workers waiting on
        # one pacer, reading the slot, sleeping, and only then advancing
        # it would hand every concurrent caller the same slot and let the
        # rate be exceeded by the concurrency.
        #
        # Anchored off max(now, scheduled) so a slow consolidate() that
        # overran its slot doesn't build up a debt the pacer then tries
        # to "catch up" by never sleeping.
        slot = now if self._next is None else max(now, self._next)
        self._next = slot + self._interval
        delay = slot - now
        if delay > 0:
            await self._sleep(delay)


def _classify(result: engine.ConsolidationResult) -> str:
    """Collapse a run's per-decision outcomes into one coarse label for
    the metric. Priority mirrors engine._summarize:
    assert > link > conflict > flag > noop."""
    outcomes = {d["outcome"] for d in result.decisions}
    if "auto_assert" in outcomes:
        return "asserted"
    if "auto_link" in outcomes:
        return "linked"
    if "conflict" in outcomes:
        return "conflict"
    if "flag" in outcomes:
        return "flagged"
    return "noop"


# Both rotation queries order by the bare property, with IS NOT NULL, so
# the planner walks the range index on last_consolidated_at in order and
# stops at the LIMIT: ~3,000 db hits for a page of 1,000 on prod's 4.8M
# companies. They used to ORDER BY coalesce(..., 1970) so never-swept
# entities came first; an expression can't use the index, so every page
# was two full label scans (14.4M db hits and ~2.8 s each), which became
# the sweep's bottleneck once it ran several entities at once. The
# never-swept are found separately and less often: see _page_unstamped.
async def _page_stalest(
    driver: AsyncDriver, database: str, label: str, key: str, page_size: int
) -> list[str]:
    query = (
        f"MATCH (n:{label}) "
        "WHERE n.last_consolidated_at IS NOT NULL AND n.name IS NOT NULL "
        f"RETURN n.{key} AS id "
        "ORDER BY n.last_consolidated_at ASC "
        "LIMIT $page"
    )
    async with driver.session(database=database) as session:
        result = await session.run(query, page=page_size)
        return [record["id"] async for record in result if record["id"] is not None]


async def _page_unstamped(
    driver: AsyncDriver, database: str, label: str, key: str, page_size: int
) -> list[str]:
    """Entities never swept at all: the stalest there are, and invisible
    to the index. A label scan, but one that stops at the LIMIT, so it is
    cheap exactly when it matters (after a reset or a big import, when
    they are everywhere) and a full scan only when there are few."""
    query = (
        f"MATCH (n:{label}) "
        "WHERE n.last_consolidated_at IS NULL AND n.name IS NOT NULL "
        f"RETURN n.{key} AS id "
        "LIMIT $page"
    )
    async with driver.session(database=database) as session:
        result = await session.run(query, page=page_size)
        return [record["id"] async for record in result if record["id"] is not None]


async def _measure_lag(driver: AsyncDriver, database: str, label: str) -> float:
    """now - oldest last_consolidated_at, in seconds, off the index (the
    first entry in index order). Computed in Cypher so we don't have to
    marshal Neo4j datetimes into Python. Never-swept entities are not in
    the index; sweep_label reports them itself."""
    query = (
        f"MATCH (n:{label}) "
        "WHERE n.last_consolidated_at IS NOT NULL AND n.name IS NOT NULL "
        "WITH n ORDER BY n.last_consolidated_at ASC LIMIT 1 "
        "RETURN duration.inSeconds(n.last_consolidated_at, datetime()).seconds AS lag"
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


#: Failures of the store, not of the entity. ServiceUnavailable and
#: SessionExpired are Neo4j down or moving (a node drain reschedules it);
#: TransientError is Neo4j shedding load -- lock timeouts, memory-pool
#: exhaustion. The same entity succeeds once the store is back.
_STORE_FAILURES = (ServiceUnavailable, SessionExpired, TransientError)

#: Outcome of an entity left unstamped because the store failed.
RETRY = "retry"

#: Deadlocks are the one store failure retried in place. With several
#: entities in flight, two duplicates of one company are often evaluated
#: at once and lock the same pair of nodes in opposite order; Neo4j kills
#: one transaction and says to retry it. Backing the whole label off for
#: SWEEP_EMPTY_BACKOFF_SEC for that would throw away the concurrency.
#: Every other store failure (Neo4j down, memory pool exhausted) still
#: backs off: retrying those in place only adds load to a struggling store.
_DEADLOCK = "Neo.TransientError.Transaction.DeadlockDetected"
_DEADLOCK_ATTEMPTS = 3


def _is_deadlock(exc: BaseException) -> bool:
    return isinstance(exc, TransientError) and getattr(exc, "code", None) == _DEADLOCK


async def _sweep_one(
    driver: AsyncDriver, database: str, label: str, key: str, entity_id: str
) -> str:
    """Re-consolidate one entity (match-only, GDS-excluded) and stamp the
    rotation cursor. Returns the coarse outcome for the metric
    ("merged"/"flagged"/.../"error"/"retry").

    Only a completed evaluation is stamped. The stamp is what the next
    page -- and the next pod, after a restart -- reads as "done", so a
    stamp on anything else skips the entity for a whole rotation:

      * a poison entity (the rules themselves raise) IS stamped, or it
        would sit at the head of every page forever;
      * a store failure is NOT: the entity is fine, Neo4j was not. It is
        returned as RETRY and stays the stalest, so it comes first again;
      * cancellation -- SIGTERM when the node drains -- propagates without
        stamping, so the entity interrupted mid-evaluation is the first
        one swept after the restart.
    """
    attempt = 0
    while True:
        attempt += 1
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
            break
        except _STORE_FAILURES as exc:
            if _is_deadlock(exc) and attempt < _DEADLOCK_ATTEMPTS:
                # Consolidation is MERGE-based and re-runnable; jitter so
                # the two deadlocked evaluations don't collide again.
                await asyncio.sleep(random.uniform(0.2, 1.0) * attempt)
                continue
            logger.warning(
                "sweeper[{label}]: Neo4j unavailable consolidating {id} ({err}); "
                "leaving it unstamped",
                label=label, id=entity_id, err=repr(exc),
            )
            return RETRY
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "sweeper[{label}]: consolidate failed for {id} ({err}); continuing",
                label=label, id=entity_id, err=repr(exc),
            )
            outcome = "error"
            break
    try:
        await _stamp(driver, database, label, key, entity_id)
    # If the stamp fails the entity stays the stalest and is retried on
    # the next page -- acceptable, just log it.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "sweeper[{label}]: stamp failed for {id} ({err})",
            label=label, id=entity_id, err=repr(exc),
        )
    return outcome


async def sweep_label(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    driver: AsyncDriver,
    database: str,
    label: str,
    *,
    stop_event: asyncio.Event,
    page_size: int,
    rate: float,
    empty_backoff_s: float,
    concurrency: int = 1,
    slots: asyncio.Semaphore | None = None,
    unstamped_scan_s: float = 300.0,
) -> None:
    """Forever: page the stalest entities for ``label`` and re-consolidate
    them, up to ``concurrency`` at once and rate-limited, stamping the
    rotation cursor as we go.

    ``slots`` bounds how many entities are in flight across ALL labels:
    run() passes one semaphore to every label so SWEEP_CONCURRENCY is the
    process-wide ceiling, not a per-label one. Without it (tests, a single
    label) the label gets a semaphore of its own."""
    key = id_key_for(label)
    pacer = _Pacer(rate)
    if slots is None:
        slots = asyncio.Semaphore(concurrency)
    SWEEP_RATE.labels(label=label).set(rate)
    logger.info(
        "sweeper[{label}]: starting (key={key}, page={page}, rate={rate}/s, "
        "concurrency={concurrency})",
        label=label, key=key, page=page_size, rate=rate, concurrency=concurrency,
    )
    # Never-swept entities go first, as they always did. Looking for them
    # is the one full scan left, so it runs when due rather than per page:
    # at once on start, again whenever the last look filled a whole page
    # (there may be more), otherwise every unstamped_scan_s.
    unstamped_due = 0.0
    while not stop_event.is_set():
        try:
            ids: list[str] = []
            if time.monotonic() >= unstamped_due:
                ids = await _page_unstamped(driver, database, label, key, page_size)
                if len(ids) < page_size:
                    unstamped_due = time.monotonic() + unstamped_scan_s
            if ids:
                # Something has never been swept, so the rotation is as far
                # behind as it can be: now - 1970, what the gauge read when
                # the query coalesced nulls to 1970.
                ROTATION_LAG.labels(label=label).set(time.time())
            else:
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

        completed = await _sweep_page(
            driver, database, label, key, ids,
            pacer=pacer, slots=slots, stop_event=stop_event,
            concurrency=concurrency,
        )
        if not completed:
            # A store failure: the rest of the page would fail the same
            # way. Back off and re-page: the unstamped entities are still
            # the stalest, so the new page starts with them.
            await _interruptible_sleep(stop_event, empty_backoff_s)


async def _sweep_page(  # pylint: disable=too-many-arguments
    driver: AsyncDriver,
    database: str,
    label: str,
    key: str,
    ids: list[str],
    *,
    pacer: _Pacer,
    slots: asyncio.Semaphore,
    stop_event: asyncio.Event,
    concurrency: int,
) -> bool:
    """Re-consolidate one page with up to ``concurrency`` workers pulling
    ids off it. Returns False if a store failure means the page should be
    abandoned; the workers stop taking new ids at that point and the ones
    already running finish.

    The whole page finishes before the next is fetched, so an entity is
    never evaluated twice at once: anything still running is unstamped
    and would otherwise come back as the stalest on the next page."""
    pending = iter(ids)
    store_failed = False

    async def worker() -> None:
        nonlocal store_failed
        while not stop_event.is_set() and not store_failed:
            entity_id = next(pending, None)
            if entity_id is None:
                return
            # Pace BEFORE taking a slot: a worker sleeping out the label's
            # rate must not hold one of the process-wide slots.
            await pacer.wait()
            if stop_event.is_set() or store_failed:
                return
            async with slots:
                outcome = await _sweep_one(driver, database, label, key, entity_id)
            SWEEP_ENTITIES.labels(label=label, outcome=outcome).inc()
            if outcome == RETRY:
                store_failed = True

    # TaskGroup, not gather: if a worker dies unexpectedly the others are
    # cancelled with it instead of running on unobserved.
    async with asyncio.TaskGroup() as group:
        for _ in range(max(1, min(concurrency, len(ids)))):
            group.create_task(worker())
    return not store_failed


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

    # One pool of slots for every label: SWEEP_CONCURRENCY is the most the
    # sweep will have in flight against Neo4j, whatever the label mix.
    slots = asyncio.Semaphore(config.concurrency)
    SWEEP_CONCURRENCY.set(config.concurrency)
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
                concurrency=config.concurrency,
                slots=slots,
                unstamped_scan_s=config.unstamped_scan_s,
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
