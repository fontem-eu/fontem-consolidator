"""Value-ordered title translation: what it selects, and when it stops.

The budget is the whole point of this backfill, so most of these are about
money: that it stops before crossing the line rather than after, that it
counts what the provider charged rather than what it guessed, and that a
failure cannot quietly consume the budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import src.consolidator.backfill_translations as backfill
from src.consolidator.backfill_translations import (
    COUNTRY_PROPERTY,
    ID_PROPERTY,
    VALUE_PROPERTY,
    Progress,
    missing_targets,
    run,
)
from src.consolidator.clients.linguistics import (
    EU_OFFICIAL_LANGS,
    LinguisticsError,
    LinguisticsUnavailable,
)

pytestmark = pytest.mark.asyncio


@dataclass
class FakeClient:
    """Stands in for fontem-linguistics, charging per call like the real one."""

    cost_per_call: float = 0.00035
    raises: Exception | None = None
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    async def translate_with_cost(self, text, source_lang, targets):
        if self.raises:
            raise self.raises
        self.calls.append((text, source_lang, len(targets)))
        return {t: f"{t}:{text}" for t in targets}, self.cost_per_call

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


@dataclass
class FakeDriver:
    """Returns rows in the order the query asked for; records writes."""

    rows: list[dict]
    written: list[dict] = field(default_factory=list)

    def session(self, database=None):
        """`database` is accepted because the real driver takes it; the fake
        holds one graph, so it has nothing to select."""
        del database
        return _FakeSession(self)

    async def close(self):
        return None


@dataclass
class _FakeSession:
    driver: FakeDriver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def run(self, query, **params):
        if "count(n)" in query:
            return _FakeResult([{"n": len(self.driver.rows)}])
        limit = params.get("limit", len(self.driver.rows))
        return _FakeResult([{"props": p} for p in self.driver.rows[:limit]])


@dataclass
class _FakeResult:
    rows: list[dict]

    def __aiter__(self):
        async def gen():
            for r in self.rows:
                yield r
        return gen()

    async def single(self):
        return self.rows[0] if self.rows else None


def _contract(notice_id: str, value: float, title: str = "Roboty budowlane") -> dict:
    return {"ted_notice_id": notice_id, "value_eur": value,
            "country": "POL", "title": title}


async def _run(rows, client, **kw):
    """Run with the linguistics client swapped for a fake."""
    original = backfill.LinguisticsClient
    backfill.LinguisticsClient = lambda **_kwargs: client
    try:
        return await run(FakeDriver(rows), "neo4j", label="Contract",
                         limit=len(rows), budget_usd=kw.pop("budget_usd", 1.0),
                         apply_changes=kw.pop("apply_changes", False), **kw)
    finally:
        backfill.LinguisticsClient = original


# ── selection ─────────────────────────────────────────────────


def test_each_label_knows_what_it_is_worth():
    """Ordering a cohesion project by value_eur would order by nothing: it
    has no such property. The money lives under a different name."""
    assert VALUE_PROPERTY["Contract"] == "value_eur"
    assert VALUE_PROPERTY["CohesionProject"] == "detail_eu_contribution"
    assert COUNTRY_PROPERTY["CohesionProject"] == "detail_country"
    assert ID_PROPERTY["CohesionProject"] == "disclosure_id"


def test_the_source_language_is_never_a_target():
    """Paying to translate a Polish title into Polish is pure waste."""
    targets = missing_targets({"title": "x"}, "pl")
    assert "pl" not in targets
    assert len(targets) == len(EU_OFFICIAL_LANGS) - 1


def test_languages_already_present_are_not_bought_twice():
    props = {"title": "x", "title_fr": "déjà", "title_de": "schon"}
    targets = missing_targets(props, "pl")
    assert "fr" not in targets and "de" not in targets
    assert len(targets) == len(EU_OFFICIAL_LANGS) - 3


# ── the budget ────────────────────────────────────────────────


async def test_it_stops_before_crossing_the_budget_not_after():
    """Checked after the call, a cap is something you notice; checked
    before, it is something you hold."""
    rows = [_contract(f"n{i}", 1000 - i) for i in range(10)]
    client = FakeClient(cost_per_call=0.10)
    progress = await _run(rows, client, budget_usd=0.25)
    # Two calls at $0.10. A third would reach $0.30, so it is not made —
    # the spend stays under the ceiling instead of stepping over it once.
    assert len(client.calls) == 2
    assert progress.spent_usd == pytest.approx(0.20)
    assert progress.spent_usd <= 0.25
    assert "budget reached" in progress.stopped_because


async def test_it_counts_what_the_provider_charged():
    rows = [_contract("n1", 100)]
    progress = await _run(rows, FakeClient(cost_per_call=0.00035))
    assert progress.spent_usd == pytest.approx(0.00035)
    assert progress.translated == 1
    assert progress.languages_written == len(EU_OFFICIAL_LANGS) - 1


async def test_the_whole_selection_can_finish_inside_the_budget():
    rows = [_contract(f"n{i}", 500 - i) for i in range(5)]
    progress = await _run(rows, FakeClient(cost_per_call=0.0001), budget_usd=5.0)
    assert progress.translated == 5
    assert progress.stopped_because == "finished the selection"


# ── order and failure ─────────────────────────────────────────


async def test_it_works_richest_first():
    """When only one call fits, it must be spent on the biggest contract."""
    rows = [_contract("big", 9_000_000, "Duży"), _contract("small", 12, "Mały")]
    client = FakeClient(cost_per_call=0.00035)
    await _run(rows, client, budget_usd=0.0005)   # room for one call only
    assert [c[0] for c in client.calls] == ["Duży"]


async def test_a_node_that_needs_nothing_costs_nothing():
    complete = {"ted_notice_id": "n1", "value_eur": 5, "country": "POL", "title": "t"}
    for code in EU_OFFICIAL_LANGS:
        if code != "pl":
            complete[f"title_{code}"] = "done"
    client = FakeClient()
    progress = await _run([complete], client)
    assert not client.calls and progress.skipped_complete == 1
    assert progress.spent_usd == 0.0


async def test_the_service_being_down_stops_the_run():
    """Unavailable means every following call would fail the same way;
    grinding through thousands of them helps nobody."""
    rows = [_contract(f"n{i}", 100 - i) for i in range(5)]
    progress = await _run(rows, FakeClient(raises=LinguisticsUnavailable("503")))
    assert progress.failed == 1
    assert "unavailable" in progress.stopped_because


async def test_one_bad_title_does_not_end_the_run():
    """A hard error on a single title is that title's problem."""
    rows = [_contract(f"n{i}", 100 - i) for i in range(3)]
    progress = await _run(rows, FakeClient(raises=LinguisticsError("bad input")))
    assert progress.failed == 3
    assert progress.stopped_because == "finished the selection"


async def test_a_dry_run_writes_nothing():
    driver = FakeDriver([_contract("n1", 100)])
    original = backfill.LinguisticsClient
    backfill.LinguisticsClient = lambda **_kw: FakeClient()
    try:
        progress = await run(driver, "neo4j", label="Contract", limit=1,
                             budget_usd=1.0, apply_changes=False)
    finally:
        backfill.LinguisticsClient = original
    assert progress.translated == 1 and not driver.written


def test_the_report_says_what_it_spent():
    text = Progress(considered=3, translated=2, spent_usd=0.0007,
                    languages_written=46, last_value=2_500_000_000).report(
                        "Contract", dry_run=True)
    assert "would translate 2 of 3" in text
    assert "$0.0007" in text
    assert "2,500,000,000 EUR" in text
