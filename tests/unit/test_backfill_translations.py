"""Value-ordered title translation: what it selects, and when it stops.

The budget is the point of this backfill, so much of this is about money:
stopping before the line rather than after, counting what was charged rather
than estimated, never paying twice for the same title, and never paying at
all for a record that already has translations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import src.consolidator.backfill_translations as backfill
from src.consolidator.backfill_translations import (
    COUNTRY_PROPERTY,
    ID_PROPERTY,
    UNDETERMINED,
    VALUE_PROPERTY,
    Progress,
    already_translated,
    plan,
    run,
    source_language,
    targets_for,
)
from src.consolidator.clients.linguistics import (
    EU_OFFICIAL_LANGS,
    LinguisticsError,
    LinguisticsUnavailable,
)

pytestmark = pytest.mark.asyncio


@dataclass
class FakeClient:
    """fontem-linguistics' batch endpoint: per-item translations, cost, error."""

    cost_per_item: float = 0.00035
    raises: Exception | None = None
    fail_titles: set[str] = field(default_factory=set)
    calls: list[tuple[list[tuple[str, str]], list[str]]] = field(default_factory=list)

    async def translate_batch_with_cost(self, items, targets):
        if self.raises:
            raise self.raises
        self.calls.append((list(items), list(targets)))
        out = []
        for text, _lang in items:
            if text in self.fail_titles:
                out.append(({}, 0.0, "NebiusError: malformed"))
            else:
                out.append(({t: f"{t}:{text}" for t in targets}, self.cost_per_item, None))
        return out

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    @property
    def titles_sent(self) -> list[str]:
        return [text for items, _t in self.calls for text, _l in items]


@dataclass
class FakeDriver:
    """Serves rows in the order the query asked for."""

    rows: list[dict]

    def session(self, database=None):
        """The real driver takes a database; this fake holds one graph."""
        del database
        return _FakeSession(self)


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
        return _FakeResult([{"props": p} for p in self.driver.rows[:params.get("limit")]])


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


def _contract(notice_id, value, title="Roboty budowlane", country="POL", **extra):
    return {"ted_notice_id": notice_id, "value_eur": value, "country": country,
            "title": title, **extra}


@pytest.fixture(name="writes")
def _writes(monkeypatch):
    """Every decision the runner would write to the graph."""
    recorded = []

    async def fake_enrich(_driver, _database, *, decision):
        recorded.append(decision)

    monkeypatch.setattr(backfill, "_enrich", fake_enrich)
    return recorded


async def _run(rows, client, monkeypatch, **kw):
    monkeypatch.setattr(backfill, "LinguisticsClient", lambda **_kw: client)
    return await run(FakeDriver(rows), "neo4j", label=kw.pop("label", "Contract"),
                     limit=len(rows), budget_usd=kw.pop("budget_usd", 1.0),
                     apply_changes=kw.pop("apply_changes", True), **kw)


# ── what gets selected ────────────────────────────────────────


def test_each_label_knows_what_it_is_worth():
    """A cohesion project has no value_eur; ordering by it orders by nothing."""
    assert VALUE_PROPERTY == {"Contract": "value_eur",
                              "CohesionProject": "detail_eu_contribution"}
    assert COUNTRY_PROPERTY["CohesionProject"] == "detail_country"
    assert ID_PROPERTY["CohesionProject"] == "disclosure_id"


def test_anything_already_translated_is_left_alone():
    """The rule is not to reprocess, so even one existing language counts."""
    assert already_translated({"title": "x", "title_fr": "déjà"})
    assert not already_translated({"title": "x"})


@pytest.mark.parametrize("country, expected", [
    ("POL", "pl"), ("DEU", "de"), ("FRA", "fr"),
    ("NOR", UNDETERMINED),     # not in the EU map: would have been "en"
    ("BEL", UNDETERMINED),     # French and Dutch
    ("MLT", UNDETERMINED),     # most Maltese TED notices are in English
    ("LUX", UNDETERMINED), ("FIN", UNDETERMINED), (None, UNDETERMINED),
])
def test_the_source_language_is_only_asserted_when_the_country_says_so(country, expected):
    assert source_language({"country": country}, "Contract") == expected


def test_an_unknown_source_asks_for_every_language():
    """With "und" the model returns the source's own entry unchanged, so no
    language can be skipped by a wrong guess — the Norwegian-without-English
    failure."""
    assert targets_for(UNDETERMINED) == list(EU_OFFICIAL_LANGS)
    assert "pl" not in targets_for("pl")
    assert len(targets_for("pl")) == len(EU_OFFICIAL_LANGS) - 1


def test_identical_titles_are_translated_once():
    """Framework agreements republish the same title; the top slice starts
    with two of them. One translation, written to both."""
    rows = [_contract("a", 900, "Construction Works", "GBR"),
            _contract("b", 900, "Construction Works", "GBR"),
            _contract("c", 800, "Roboty budowlane")]
    work, skipped = plan(rows, "Contract")
    assert skipped == 0
    assert [w.title for w in work] == ["Construction Works", "Roboty budowlane"]
    assert work[0].node_ids == ["a", "b"]


# ── the money ─────────────────────────────────────────────────


@pytest.mark.usefixtures("writes")
async def test_the_first_title_prices_the_rest(monkeypatch):
    """A round priced from the seed estimate would stake 32 titles on the
    guess; the probe measures one first."""
    rows = [_contract(f"n{i}", 1000 - i, f"title {i}") for i in range(10)]
    client = FakeClient(cost_per_item=0.10)                   # 285x the seed
    await _run(rows, client, monkeypatch, budget_usd=0.35)
    assert [len(items) for items, _t in client.calls] == [1, 2]
    assert sum(len(items) for items, _t in client.calls) * 0.10 <= 0.35


async def test_it_stops_before_crossing_the_budget(monkeypatch, writes):
    rows = [_contract(f"n{i}", 1000 - i, f"title {i}") for i in range(10)]
    client = FakeClient(cost_per_item=0.10)
    progress = await _run(rows, client, monkeypatch, budget_usd=0.25)
    assert progress.spent_usd <= 0.25
    assert len(client.titles_sent) == 2
    assert "budget reached" in progress.stopped_because
    assert len(writes) == 2


async def test_it_counts_what_was_charged(monkeypatch, writes):
    progress = await _run([_contract("n1", 100)], FakeClient(cost_per_item=0.00035),
                          monkeypatch)
    assert progress.spent_usd == pytest.approx(0.00035)
    assert progress.languages_written == len(EU_OFFICIAL_LANGS) - 1
    assert len(writes) == 1


async def test_nothing_already_translated_is_paid_for(monkeypatch, writes):
    rows = [_contract("done", 999, title_de="schon"), _contract("new", 5)]
    client = FakeClient()
    progress = await _run(rows, client, monkeypatch)
    assert client.titles_sent == ["Roboty budowlane"]
    assert progress.skipped_complete == 1
    assert [d.source_id for d in writes] == ["new"]


@pytest.mark.usefixtures("writes")
async def test_it_works_richest_first(monkeypatch):
    """When only one title fits, it must be the biggest contract's."""
    rows = [_contract("big", 9_000_000, "Duży"), _contract("small", 12, "Mały")]
    client = FakeClient(cost_per_item=0.00035)
    await _run(rows, client, monkeypatch, budget_usd=0.0005)
    assert client.titles_sent == ["Duży"]


# ── batching ──────────────────────────────────────────────────


@pytest.mark.usefixtures("writes")
async def test_a_batch_never_mixes_source_languages(monkeypatch):
    """Targets are shared across a batch, so a batch must share a source:
    a Polish title's batch cannot ask for Polish, a Norwegian one must ask
    for everything. The first title runs alone, to price the rest."""
    rows = [_contract("a", 9, "Polski", "POL"), _contract("b", 8, "Deutsch", "DEU"),
            _contract("c", 7, "Też polski", "POL"), _contract("d", 6, "Norsk", "NOR")]
    client = FakeClient()
    await _run(rows, client, monkeypatch)
    for items, targets in client.calls:
        langs = {lang for _t, lang in items}
        assert len(langs) == 1, f"mixed batch: {items}"
        (lang,) = langs
        assert targets == backfill.targets_for(lang)
    assert client.calls[0][0] == [("Polski", "pl")]            # the probe
    after_probe = client.calls[1:]
    assert len(after_probe) == 3                               # pl, de, und
    assert sorted(t for t, _ in after_probe[0][0] + after_probe[1][0] + after_probe[2][0]) \
        == ["Deutsch", "Norsk", "Też polski"]


async def test_a_duplicate_title_is_written_to_every_node(monkeypatch, writes):
    rows = [_contract("a", 9, "Same", "GBR"), _contract("b", 9, "Same", "GBR")]
    client = FakeClient()
    progress = await _run(rows, client, monkeypatch)
    assert client.titles_sent == ["Same"]
    assert sorted(d.source_id for d in writes) == ["a", "b"]
    assert progress.translated == 2 and progress.spent_usd == pytest.approx(0.00035)


async def test_an_undetermined_language_is_not_written_as_a_claim(monkeypatch, writes):
    await _run([_contract("n", 9, "Anskaffelse", "NOR")], FakeClient(), monkeypatch)
    assert writes[0].details["source_lang"] is None
    await _run([_contract("p", 9, "Roboty", "POL")], FakeClient(), monkeypatch)
    assert writes[1].details["source_lang"] == "pl"


# ── failure ───────────────────────────────────────────────────


async def test_one_failed_title_does_not_cost_the_others(monkeypatch, writes):
    rows = [_contract("a", 9, "good one"), _contract("b", 8, "bad one"),
            _contract("c", 7, "good two")]
    progress = await _run(rows, FakeClient(fail_titles={"bad one"}), monkeypatch)
    assert progress.failed == 1
    assert sorted(d.source_id for d in writes) == ["a", "c"]


async def test_the_service_being_down_stops_the_run(monkeypatch, writes):
    progress = await _run([_contract("n", 9)], FakeClient(raises=LinguisticsUnavailable("503")),
                          monkeypatch)
    assert "unavailable" in progress.stopped_because and not writes


async def test_a_hard_error_skips_the_round_not_the_run(monkeypatch, writes):
    progress = await _run([_contract("n", 9)], FakeClient(raises=LinguisticsError("400")),
                          monkeypatch)
    assert progress.failed == 1 and not writes


async def test_a_dry_run_writes_nothing(monkeypatch, writes):
    progress = await _run([_contract("n", 9)], FakeClient(), monkeypatch, apply_changes=False)
    assert progress.translated == 1 and not writes


def test_the_report_says_what_it_spent_and_skipped():
    text = Progress(considered=3, distinct_titles=2, translated=2, skipped_complete=1,
                    spent_usd=0.0007, languages_written=46,
                    last_value=2_500_000_000).report("Contract", dry_run=True)
    assert "would translate 2 of 3" in text
    assert "not reprocessed" in text
    assert "$0.0007" in text and "2,500,000,000 EUR" in text
