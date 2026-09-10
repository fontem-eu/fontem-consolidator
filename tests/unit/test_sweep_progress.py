"""Measuring a full-corpus re-consolidation pass.

The number this produces is the one someone plans around, so the ways it
can lie matter more than the ways it can crash: counting entities the
sweeper will never touch (so it never reaches 100%), or deriving a rate
from configuration rather than observation (which on prod would have
been optimistic by ~2x).
"""
# pylint: disable=protected-access
import asyncio

from src.consolidator import sweep_progress as sp


def test_population_matches_the_sweepers_own_filter():
    """The sweeper skips nameless entities (_page_stalest). Counting them
    as outstanding would leave the pass permanently short of 100% with no
    way to tell that from real remaining work."""
    assert "WHERE n.name IS NOT NULL" in sp._SWEEPABLE
    assert "WHERE n.name IS NOT NULL" in sp._COUNTS


def test_done_is_measured_against_the_pass_start_not_null():
    """`last_consolidated_at IS NOT NULL` would count the previous
    generation's work as done. Prod had 4,149,809 entities stamped by the
    consolidator whose results were just deleted -- reading those as
    complete would have reported the pass ~90% finished on day one."""
    assert "last_consolidated_at >= datetime($since)" in sp._COUNTS
    assert "IS NOT NULL" not in sp._COUNTS.split("RETURN")[1]


def test_rate_is_sampled_from_the_graph(monkeypatch):
    """Observed, not configured. The prod ceiling is 6/sec and the real
    throughput was 3.33/sec, because the limiter is a ceiling and the
    consolidation downstream is what paces the sweep."""
    seq = [(100, 10), (100, 40)]

    async def _fake_counts(_label, _since):
        return seq.pop(0)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(sp, "_counts", _fake_counts)
    monkeypatch.setattr(sp.asyncio, "sleep", _no_sleep)
    rate = asyncio.run(sp._measured_rate("Company", "2026-01-01T00:00:00Z", 30))
    assert rate == 1.0          # 30 entities in 30s


def test_a_stalled_sweeper_is_reported_not_divided_by(monkeypatch):
    """Zero progress must not become a divide-by-zero or an infinite ETA
    presented as fact -- a stopped sweeper is the thing most worth
    saying out loud."""
    async def _fake_counts(_label, _since):
        return (100, 10)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(sp, "_counts", _fake_counts)
    monkeypatch.setattr(sp.asyncio, "sleep", _no_sleep)
    asyncio.run(sp.report(["Company"], "2026-01-01T00:00:00Z", 5))


def test_human_never_claims_to_know_an_impossible_eta():
    assert sp._human(0) == "unknown"
    assert sp._human(-5) == "unknown"
    assert sp._human(86400 + 3600) == "1d 1h"


def test_counts_asks_neo4j_once_and_unpacks_both_numbers(monkeypatch):
    """One query for total and done together -- two round-trips could
    straddle a sweeper write and report done > total."""
    seen = {}

    class _Result:
        async def single(self):
            return {"total": 4600950, "done": 26806}

    class _Session:
        async def run(self, query, **kw):
            seen["query"] = query
            seen["since"] = kw.get("since")
            return _Result()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

    class _Driver:
        def session(self, **_k):
            return _Session()

    async def _driver():
        return _Driver()

    monkeypatch.setattr(sp, "get_driver", _driver)
    total, done = asyncio.run(sp._counts("Company", "2026-09-10T22:50:00Z"))
    assert (total, done) == (4600950, 26806)
    assert seen["since"] == "2026-09-10T22:50:00Z"
    assert "MATCH (n:Company)" in seen["query"]


def test_a_finished_pass_reports_complete_and_skips_sampling(monkeypatch):
    """Nothing left to do means no reason to spend a sample window
    measuring a rate nobody needs."""
    sampled = []

    async def _fake_counts(_label, _since):
        return (100, 100)

    async def _fake_rate(*_a):
        sampled.append(1)
        return 1.0

    monkeypatch.setattr(sp, "_counts", _fake_counts)
    monkeypatch.setattr(sp, "_measured_rate", _fake_rate)
    asyncio.run(sp.report(["Company"], "2026-01-01T00:00:00Z", 5))
    assert not sampled


def test_report_computes_an_eta_from_the_observed_rate(monkeypatch):
    async def _fake_counts(_label, _since):
        return (1000, 100)

    async def _fake_rate(*_a):
        return 0.5           # 900 remaining / 0.5 = 1800s

    monkeypatch.setattr(sp, "_counts", _fake_counts)
    monkeypatch.setattr(sp, "_measured_rate", _fake_rate)
    asyncio.run(sp.report(["Company"], "2026-01-01T00:00:00Z", 5))


def test_main_wires_the_cli_and_always_closes_the_driver(monkeypatch):
    """A leaked driver keeps the process alive, which turns a one-shot
    reporting Job into one that never reaches Complete."""
    closed = []
    called = {}

    async def _fake_report(labels, since, sample):
        called["args"] = (labels, since, sample)

    async def _fake_close():
        closed.append(1)

    monkeypatch.setattr(sp, "report", _fake_report)
    monkeypatch.setattr(sp, "close_driver", _fake_close)
    sp.main(["--since", "2026-09-10T22:50:00Z",
             "--labels", "Company, Authority", "--sample-seconds", "7"])
    assert called["args"] == (["Company", "Authority"],
                              "2026-09-10T22:50:00Z", 7)
    assert closed == [1]


def test_the_driver_is_closed_even_when_reporting_raises(monkeypatch):
    closed = []

    async def _boom(*_a):
        raise RuntimeError("neo4j went away")

    async def _fake_close():
        closed.append(1)

    monkeypatch.setattr(sp, "report", _boom)
    monkeypatch.setattr(sp, "close_driver", _fake_close)
    try:
        sp.main(["--since", "2026-01-01T00:00:00Z"])
    except RuntimeError:
        pass
    assert closed == [1]
