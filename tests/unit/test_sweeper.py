"""Tests for the re-consolidation sweeper.

Covers the rotation contract:
  - the page query orders stalest-first (never-stamped nodes win);
  - each swept id is re-consolidated match-only, GDS-excluded, and then
    stamped with the rotation cursor;
  - a consolidate() exception is caught, the loop continues, and the id
    is STILL stamped (a poison entity can't wedge the rotation);
  - an empty page backs off instead of hot-looping;
  - the pacer honours the configured per-label rate;
  - the per-entity counter increments with the coarse outcome.
"""
# import-outside-toplevel: sweeper pulls in prometheus collectors at
# import time; importing inside tests keeps that off module import.
# redefined-outer-name / protected-access: pytest fixture pattern +
# poking the module-level sweeper functions is the whole point here.
# pylint: disable=import-outside-toplevel,redefined-outer-name,protected-access
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.consolidator import sweeper
from src.consolidator.engine import ConsolidationResult


def _result(*outcomes: str) -> ConsolidationResult:
    return ConsolidationResult(
        run_id="r",
        entity_type="Company",
        entity_id="x",
        decisions=[{"outcome": o} for o in outcomes],
        rules_fired=len(outcomes),
    )


# --------------------------------------------------------------------------
# Fakes for the async Neo4j driver/session.
# --------------------------------------------------------------------------
class _AsyncResult:
    def __init__(self, rows: list[dict]):
        self._rows = list(rows)
        self._it = iter(())

    def __aiter__(self):
        self._it = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration as exc:  # pragma: no cover - trivial
            raise StopAsyncIteration from exc

    async def single(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, driver: "_FakeDriver"):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def run(self, query, **params):
        self._driver.queries.append((query, params))
        return self._driver.responder(query, params)


class _FakeDriver:
    """Records every query; ``responder`` maps a query to an _AsyncResult."""

    def __init__(self, responder):
        self.responder = responder
        self.queries: list[tuple[str, dict]] = []

    def session(self, database=None):  # noqa: D401 - mimics AsyncDriver
        assert database == "neo4j"
        return _FakeSession(self)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def test_config_defaults_and_rates():
    with patch.dict("os.environ", {}, clear=True):
        cfg = sweeper.SweeperConfig.from_env()
    assert cfg.labels == ["Company", "Authority"]
    assert cfg.page_size == 200
    assert cfg.metrics_port == 9100
    assert cfg.rates == {"Company": 6.0, "Authority": 2.0}


def test_config_rate_overrides():
    with patch.dict(
        "os.environ",
        {"SWEEP_COMPANY_RATE_PER_SEC": "10", "SWEEP_AUTHORITY_RATE_PER_SEC": "1"},
        clear=True,
    ):
        cfg = sweeper.SweeperConfig.from_env()
    assert cfg.rates == {"Company": 10.0, "Authority": 1.0}


# --------------------------------------------------------------------------
# Query shape
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_page_stalest_orders_oldest_first():
    driver = _FakeDriver(
        lambda q, p: _AsyncResult([{"id": "gmr-1"}, {"id": "gmr-2"}])
    )
    ids = await sweeper._page_stalest(driver, "neo4j", "Company", "gmr_id", 200)
    assert ids == ["gmr-1", "gmr-2"]
    query, params = driver.queries[0]
    assert "RETURN n.gmr_id AS id" in query
    assert "ORDER BY coalesce(n.last_consolidated_at, datetime('1970-01-01')) ASC" in query
    assert "n.name IS NOT NULL" in query
    assert params == {"page": 200}


@pytest.mark.asyncio
async def test_stamp_sets_cursor():
    driver = _FakeDriver(lambda q, p: _AsyncResult([]))
    await sweeper._stamp(driver, "neo4j", "Authority", "authority_id", "auth-9")
    query, params = driver.queries[0]
    assert "SET n.last_consolidated_at = datetime()" in query
    assert "(n:Authority {authority_id: $id})" in query
    assert params == {"id": "auth-9"}


@pytest.mark.asyncio
async def test_measure_lag_reads_seconds():
    driver = _FakeDriver(lambda q, p: _AsyncResult([{"lag": 4242}]))
    lag = await sweeper._measure_lag(driver, "neo4j", "Company")
    assert lag == 4242.0


# --------------------------------------------------------------------------
# Outcome classification
# --------------------------------------------------------------------------
def test_classify_priority():
    assert sweeper._classify(_result("noop", "flag", "auto_merge")) == "merged"
    assert sweeper._classify(_result("flag", "auto_link")) == "linked"
    assert sweeper._classify(_result("conflict", "flag")) == "conflict"
    assert sweeper._classify(_result("flag")) == "flagged"
    assert sweeper._classify(_result("noop")) == "noop"
    assert sweeper._classify(_result()) == "noop"


# --------------------------------------------------------------------------
# Pacer / rate limiting
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pacer_honours_rate():
    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    now = {"t": 0.0}
    pacer = sweeper._Pacer(2.0, sleep=fake_sleep, clock=lambda: now["t"])
    # First acquisition: no wait, next slot armed at +0.5.
    await pacer.wait()
    # Second acquisition at the same instant must sleep the full interval.
    await pacer.wait()
    assert sleeps == [0.5]


@pytest.mark.asyncio
async def test_pacer_zero_rate_never_sleeps():
    sleeps: list[float] = []

    async def fake_sleep(secs):  # pragma: no cover - must never run
        sleeps.append(secs)

    pacer = sweeper._Pacer(0.0, sleep=fake_sleep, clock=lambda: 0.0)
    await pacer.wait()
    await pacer.wait()
    assert not sleeps


# --------------------------------------------------------------------------
# Sweep loop
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sweep_loop_consolidates_and_stamps_each_id():
    stop = asyncio.Event()
    ids = ["a", "b", "c"]
    calls = {"page": 0}

    async def fake_page(*_a, **_k):
        calls["page"] += 1
        if calls["page"] == 1:
            return list(ids)
        stop.set()
        return []

    consolidate = AsyncMock(return_value=_result("noop"))
    stamp = AsyncMock()

    with patch.object(sweeper, "_page_stalest", fake_page), patch.object(
        sweeper, "_measure_lag", AsyncMock(return_value=0.0)
    ), patch.object(sweeper, "_stamp", stamp), patch.object(
        sweeper.engine, "consolidate", consolidate
    ):
        await sweeper.sweep_label(
            AsyncMock(), "neo4j", "Company",
            stop_event=stop, page_size=200, rate=0.0, empty_backoff_s=0.0,
        )

    # each id consolidated match-only, gds-excluded, triggered_by sweeper
    assert consolidate.await_count == 3
    for call, entity_id in zip(consolidate.await_args_list, ids):
        assert call.kwargs["entity_type"] == "Company"
        assert call.kwargs["entity_id"] == entity_id
        assert call.kwargs["mode"] == "match_only"
        assert call.kwargs["exclude_rule_prefix"] == "gds_"
        assert call.kwargs["triggered_by"] == "sweeper"
    # every id stamped
    assert [c.args[4] for c in stamp.await_args_list] == ids


@pytest.mark.asyncio
async def test_sweep_loop_survives_consolidate_error_and_still_stamps():
    stop = asyncio.Event()
    calls = {"page": 0}

    async def fake_page(*_a, **_k):
        calls["page"] += 1
        if calls["page"] == 1:
            return ["boom"]
        stop.set()
        return []

    consolidate = AsyncMock(side_effect=RuntimeError("poison entity"))
    stamp = AsyncMock()

    with patch.object(sweeper, "_page_stalest", fake_page), patch.object(
        sweeper, "_measure_lag", AsyncMock(return_value=0.0)
    ), patch.object(sweeper, "_stamp", stamp), patch.object(
        sweeper.engine, "consolidate", consolidate
    ):
        # Must not raise — the error is swallowed and the loop moves on.
        await sweeper.sweep_label(
            AsyncMock(), "neo4j", "Company",
            stop_event=stop, page_size=200, rate=0.0, empty_backoff_s=0.0,
        )

    # The poison id was still stamped so the cursor advances past it.
    assert stamp.await_count == 1
    assert stamp.await_args_list[0].args[4] == "boom"


@pytest.mark.asyncio
async def test_sweep_loop_backs_off_on_empty_page():
    stop = asyncio.Event()
    backoff = AsyncMock(side_effect=lambda ev, s: stop.set())
    consolidate = AsyncMock()

    with patch.object(
        sweeper, "_page_stalest", AsyncMock(return_value=[])
    ), patch.object(sweeper, "_measure_lag", AsyncMock(return_value=0.0)), patch.object(
        sweeper, "_interruptible_sleep", backoff
    ), patch.object(sweeper.engine, "consolidate", consolidate):
        await sweeper.sweep_label(
            AsyncMock(), "neo4j", "Company",
            stop_event=stop, page_size=200, rate=0.0, empty_backoff_s=30.0,
        )

    backoff.assert_awaited()
    assert backoff.await_args_list[0].args[1] == 30.0
    consolidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_loop_increments_outcome_counter():
    from prometheus_client import REGISTRY

    def _val(outcome):
        return REGISTRY.get_sample_value(
            "consolidator_sweep_entities_total",
            {"label": "Company", "outcome": outcome},
        ) or 0.0

    before = _val("flagged")
    stop = asyncio.Event()
    calls = {"page": 0}

    async def fake_page(*_a, **_k):
        calls["page"] += 1
        if calls["page"] == 1:
            return ["z"]
        stop.set()
        return []

    with patch.object(sweeper, "_page_stalest", fake_page), patch.object(
        sweeper, "_measure_lag", AsyncMock(return_value=0.0)
    ), patch.object(sweeper, "_stamp", AsyncMock()), patch.object(
        sweeper.engine, "consolidate", AsyncMock(return_value=_result("flag"))
    ):
        await sweeper.sweep_label(
            AsyncMock(), "neo4j", "Company",
            stop_event=stop, page_size=200, rate=0.0, empty_backoff_s=0.0,
        )

    assert _val("flagged") == before + 1.0
    # rate gauge published for the label
    assert REGISTRY.get_sample_value(
        "consolidator_sweep_rate", {"label": "Company"}
    ) == 0.0
