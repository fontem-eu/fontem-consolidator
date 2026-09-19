"""Never-swept entities: found by their own scan, less often than pages.

They are the only entities the last_consolidated_at index can't find, so
looking for them is a full label scan. The loop must still sweep them
first -- that is the rotation contract -- but not pay that scan per page.
"""
# protected-access: the loop's helpers are module internals.
# pylint: disable=protected-access
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator import sweeper
from src.consolidator.engine import ConsolidationResult


def _result() -> ConsolidationResult:
    return ConsolidationResult(run_id="r", entity_type="Company", entity_id="x",
                               decisions=[], rules_fired=0)


async def _loop(unstamped_pages, stalest_pages, *, page_size=3,  # pylint: disable=too-many-locals
                unstamped_scan_s=300.0, clock=None):
    """Run sweep_label until both page sources are exhausted; return what
    was consolidated, in order, and how many times each source was asked."""
    stop = asyncio.Event()
    unstamped = list(unstamped_pages)
    stalest = list(stalest_pages)
    calls = {"unstamped": 0, "stalest": 0, "lag": 0}

    async def page_unstamped(*_a, **_k):
        calls["unstamped"] += 1
        return unstamped.pop(0) if unstamped else []

    async def page_stalest(*_a, **_k):
        calls["stalest"] += 1
        if stalest:
            return stalest.pop(0)
        stop.set()
        return []

    async def measure_lag(*_a, **_k):
        calls["lag"] += 1
        return 7.0

    seen: list[str] = []

    async def consolidate(*_a, **kw):
        seen.append(kw["entity_id"])
        return _result()

    patches = [
        patch.object(sweeper, "_page_unstamped", page_unstamped),
        patch.object(sweeper, "_page_stalest", page_stalest),
        patch.object(sweeper, "_measure_lag", measure_lag),
        patch.object(sweeper, "_stamp", AsyncMock()),
        patch.object(sweeper.engine, "consolidate", consolidate),
        patch.object(sweeper, "_interruptible_sleep", AsyncMock()),
    ]
    if clock is not None:
        patches.append(patch.object(sweeper.time, "monotonic", clock))
    for p in patches:
        p.start()
    try:
        await sweeper.sweep_label(
            AsyncMock(), "neo4j", "Company", stop_event=stop, page_size=page_size,
            rate=0.0, empty_backoff_s=0.0, unstamped_scan_s=unstamped_scan_s,
        )
    finally:
        for p in reversed(patches):
            p.stop()
    return seen, calls


@pytest.mark.asyncio
async def test_never_swept_go_first_then_the_index():
    seen, _ = await _loop([["new-1"]], [["old-1", "old-2"]])
    assert seen == ["new-1", "old-1", "old-2"]


@pytest.mark.asyncio
async def test_the_scan_is_not_repeated_per_page_once_drained():
    """A short page of never-swept means they're all found: the next look
    waits unstamped_scan_s, and meanwhile every page comes off the index."""
    now = {"t": 1000.0}
    _, calls = await _loop([[]], [["a"], ["b"], ["c"], ["d"]], clock=lambda: now["t"])
    assert calls["unstamped"] == 1
    assert calls["stalest"] == 5  # four pages + the empty one that stops the test


@pytest.mark.asyncio
async def test_a_full_page_of_never_swept_is_looked_at_again_straight_away():
    """After a reset or a big import there are more than a page of them."""
    seen, calls = await _loop([["n1", "n2", "n3"], ["n4"]], [["old"]], page_size=3)
    assert seen == ["n1", "n2", "n3", "n4", "old"]
    assert calls["unstamped"] == 2


@pytest.mark.asyncio
async def test_the_scan_comes_back_when_due():
    now = {"t": 1000.0}

    def clock():
        now["t"] += 200.0  # each look at the clock is 200 s later
        return now["t"]

    _, calls = await _loop([[], ["late"], []], [["a"], ["b"], ["c"]],
                           unstamped_scan_s=300.0, clock=clock)
    assert calls["unstamped"] >= 2


@pytest.mark.asyncio
async def test_lag_reads_as_far_behind_as_possible_while_never_swept_wait():
    """While never-swept entities are waiting the gauge reads now - 1970, as
    it did when nulls were coalesced to 1970, and the index isn't asked."""
    gauge = MagicMock()
    with patch.object(sweeper, "ROTATION_LAG", gauge), \
         patch.object(sweeper.time, "time", return_value=1.7e9):
        _, calls = await _loop([["new-1"]], [["old-1"]])
    values = [c.args[0] for c in gauge.labels.return_value.set.call_args_list]
    assert values[0] == 1.7e9          # the never-swept page
    assert values[1:] == [7.0, 7.0]    # index pages: old-1, then the empty stop page
    assert calls["lag"] == calls["stalest"]


def test_scan_interval_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SWEEP_UNSTAMPED_SCAN_SEC", "60")
    assert sweeper.SweeperConfig.from_env().unstamped_scan_s == 60.0
    monkeypatch.delenv("SWEEP_UNSTAMPED_SCAN_SEC")
    assert sweeper.SweeperConfig.from_env().unstamped_scan_s == 300.0
