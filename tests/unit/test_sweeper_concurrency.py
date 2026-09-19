"""SWEEP_CONCURRENCY: several entities in flight, bounded, rate still honoured.

The contract the concurrency must not break:
  - never more than SWEEP_CONCURRENCY entities being evaluated at once,
    counting every label together;
  - every id on a page evaluated exactly once;
  - the per-label pacer still caps the rate when several workers share it;
  - a store failure stops the page from taking new ids and backs off;
  - a deadlock is retried in place, anything else from the store is not.
"""
# protected-access: the page runner and pacer are module internals.
# pylint: disable=protected-access
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.consolidator import sweeper
from src.consolidator.engine import ConsolidationResult


def _result() -> ConsolidationResult:
    return ConsolidationResult(run_id="r", entity_type="Company", entity_id="x",
                               decisions=[], rules_fired=0)


def _deadlock():
    return Neo4jError._hydrate_neo4j(
        code="Neo.TransientError.Transaction.DeadlockDetected", message="deadlock",
    )


class _InFlight:
    """A consolidate() stand-in that records how many calls overlap."""

    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self.now = 0
        self.peak = 0
        self.seen: list[str] = []

    async def __call__(self, *_a, **kw):
        self.now += 1
        self.peak = max(self.peak, self.now)
        self.seen.append(kw["entity_id"])
        await asyncio.sleep(self.delay)
        self.now -= 1
        return _result()


async def _run_page(ids, *, concurrency, slots=None, rate=0.0, label="Company"):
    return await sweeper._sweep_page(
        AsyncMock(), "neo4j", label, "gmr_id", ids,
        pacer=sweeper._Pacer(rate),
        slots=slots or asyncio.Semaphore(concurrency),
        stop_event=asyncio.Event(), concurrency=concurrency,
    )


async def _page(ids, *, concurrency, consolidate, slots=None, rate=0.0,  # pylint: disable=too-many-arguments
                label="Company"):
    with patch.object(sweeper.engine, "consolidate", consolidate), \
         patch.object(sweeper, "_stamp", AsyncMock()):
        return await _run_page(ids, concurrency=concurrency, slots=slots, rate=rate, label=label)


@pytest.mark.asyncio
async def test_page_runs_up_to_concurrency_at_once_and_each_id_once():
    fake = _InFlight()
    ids = [f"id{i}" for i in range(10)]
    assert await _page(ids, concurrency=4, consolidate=fake) is True
    assert fake.peak == 4
    assert sorted(fake.seen) == sorted(ids)


@pytest.mark.asyncio
async def test_concurrency_one_is_the_old_serial_sweep():
    fake = _InFlight()
    ids = ["a", "b", "c"]
    await _page(ids, concurrency=1, consolidate=fake)
    assert fake.peak == 1
    assert fake.seen == ids


@pytest.mark.asyncio
async def test_the_ceiling_is_shared_across_labels():
    """run() hands every label the same semaphore: 4 means 4 in total."""
    fake = _InFlight()
    slots = asyncio.Semaphore(2)
    # Patched once around both pages: two overlapping patch contexts would
    # restore each other's mocks in the wrong order and leak one.
    with patch.object(sweeper.engine, "consolidate", fake), \
         patch.object(sweeper, "_stamp", AsyncMock()):
        await asyncio.gather(
            _run_page([f"c{i}" for i in range(6)], concurrency=2, slots=slots),
            _run_page([f"a{i}" for i in range(6)], concurrency=2, slots=slots, label="Authority"),
        )
    assert fake.peak == 2
    assert len(fake.seen) == 12


@pytest.mark.asyncio
async def test_a_store_failure_stops_the_page_taking_new_ids():
    calls: list[str] = []

    async def consolidate(*_a, **kw):
        calls.append(kw["entity_id"])
        await asyncio.sleep(0)
        raise ServiceUnavailable("neo4j restarting")

    ids = [f"id{i}" for i in range(20)]
    assert await _page(ids, concurrency=4, consolidate=consolidate) is False
    # The workers already running finish; nobody starts the rest.
    assert len(calls) <= 4


@pytest.mark.asyncio
async def test_the_pacer_holds_the_rate_with_concurrent_callers():
    """Slots are booked before sleeping, so N concurrent callers get N
    distinct slots instead of N copies of the same one."""
    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    pacer = sweeper._Pacer(4.0, sleep=fake_sleep, clock=lambda: 0.0)
    await asyncio.gather(*(pacer.wait() for _ in range(4)))
    assert sorted(sleeps) == [0.25, 0.5, 0.75]


@pytest.mark.asyncio
async def test_a_deadlock_is_retried_in_place_and_then_stamped():
    consolidate = AsyncMock(side_effect=[_deadlock(), _result()])
    stamp = AsyncMock()
    with patch.object(sweeper.engine, "consolidate", consolidate), \
         patch.object(sweeper, "_stamp", stamp), \
         patch.object(sweeper.asyncio, "sleep", AsyncMock()):
        outcome = await sweeper._sweep_one(AsyncMock(), "neo4j", "Company", "gmr_id", "a")
    assert outcome == "noop"
    assert consolidate.await_count == 2
    stamp.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_deadlock_that_keeps_happening_becomes_a_retry():
    consolidate = AsyncMock(side_effect=[_deadlock()] * sweeper._DEADLOCK_ATTEMPTS)
    stamp = AsyncMock()
    with patch.object(sweeper.engine, "consolidate", consolidate), \
         patch.object(sweeper, "_stamp", stamp), \
         patch.object(sweeper.asyncio, "sleep", AsyncMock()):
        outcome = await sweeper._sweep_one(AsyncMock(), "neo4j", "Company", "gmr_id", "a")
    assert outcome == sweeper.RETRY
    assert consolidate.await_count == sweeper._DEADLOCK_ATTEMPTS
    stamp.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_transient_errors_are_not_retried_in_place():
    """Memory-pool exhaustion is Neo4j shedding load: back off, don't hammer."""
    pool = Neo4jError._hydrate_neo4j(
        code="Neo.TransientError.General.MemoryPoolOutOfMemoryError", message="pool",
    )
    consolidate = AsyncMock(side_effect=pool)
    with patch.object(sweeper.engine, "consolidate", consolidate), \
         patch.object(sweeper, "_stamp", AsyncMock()):
        outcome = await sweeper._sweep_one(AsyncMock(), "neo4j", "Company", "gmr_id", "a")
    assert outcome == sweeper.RETRY
    assert consolidate.await_count == 1


def test_concurrency_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SWEEP_CONCURRENCY", "4")
    assert sweeper.SweeperConfig.from_env().concurrency == 4


@pytest.mark.parametrize("raw", ["0", "-3"])
def test_concurrency_is_never_below_one(monkeypatch, raw):
    monkeypatch.setenv("SWEEP_CONCURRENCY", raw)
    assert sweeper.SweeperConfig.from_env().concurrency == 1


def test_concurrency_defaults_to_one(monkeypatch):
    monkeypatch.delenv("SWEEP_CONCURRENCY", raising=False)
    assert sweeper.SweeperConfig.from_env().concurrency == 1
