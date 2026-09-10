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
