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
    CANONICAL_FILTER,
    ID_PROPERTY,
    ROUND_TIMEOUT_S,
    SEED_COST_USD,
    UNDETERMINED,
    VALUE_PROPERTY,
    Progress,
    already_translated,
    has_no_words,
    plan,
    run,
    source_language,
    targets_for,
    translated_from_the_wrong_language,
)
from src.consolidator.clients.linguistics import (
    EU_OFFICIAL_LANGS,
    LinguisticsError,
    LinguisticsUnavailable,
)

@dataclass
class FakeClient:
    """fontem-linguistics' batch endpoint: per-item translations, cost, error."""

    cost_per_item: float = 0.00035
    raises: Exception | None = None
    fail_titles: set[str] = field(default_factory=set)
    calls: list[tuple[list[tuple[str, str]], list[str]]] = field(default_factory=list)
    #: /detect: title -> what the model answers (default "pl"; None = no answer).
    detections: dict[str, str | None] = field(default_factory=dict)
    detect_raises: Exception | None = None
    detect_calls: list[list[str]] = field(default_factory=list)

    async def detect(self, texts):
        if self.detect_raises:
            raise self.detect_raises
        self.detect_calls.append(list(texts))
        return ([self.detections.get(t, "pl") for t in texts],
                "nebius:google/gemma-3-27b-it", 0.00001 * len(texts))

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
    queries: list[str] = field(default_factory=list)
    writes: list[tuple[str, dict]] = field(default_factory=list)

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
        self.driver.queries.append(query)
        if " SET " in query:
            self.driver.writes.append((query, params))
            return _FakeResult([])
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


#: What the notice states, for the countries these tests use. The runner no
#: longer reads the country; the rows carry the notice's statement instead.
_STATED = {"POL": "pl", "DEU": "de", "FRA": "fr", "GBR": "en", "NOR": "no"}


def _contract(notice_id, value, title="Roboty budowlane", country="POL", **extra):
    """A contract row. ``title_lang`` defaults to what its notice would state;
    pass ``title_lang=None`` for one that states nothing."""
    row = {"ted_notice_id": notice_id, "value_eur": value, "country": country,
           "title": title, "title_lang": _STATED.get(country), **extra}
    return {k: v for k, v in row.items() if v is not None}


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
    assert ID_PROPERTY["CohesionProject"] == "disclosure_id"


@pytest.mark.asyncio
async def test_only_canonically_keyed_contracts_are_selected_or_counted():
    """A contract keyed by its OJ S reference is the duplicate of one keyed
    by its publication number, and is queued to be merged into it."""
    driver = FakeDriver([_contract("n", 9)])
    await backfill.select_by_value(driver, "neo4j", "Contract", 5)
    await backfill.count_candidates(driver, "neo4j", "Contract")
    assert all("NOT n.ted_notice_id CONTAINS '/S '" in q for q in driver.queries)
    assert CANONICAL_FILTER["CohesionProject"] == ""


def test_anything_already_translated_is_left_alone():
    """The rule is not to reprocess, so even one existing language counts."""
    assert already_translated({"title": "x", "title_fr": "déjà"})
    assert not already_translated({"title": "x"})


@pytest.mark.parametrize("country", ["POL", "DEU", "BEL", "NOR", None])
def test_the_buyers_country_never_decides_the_language(country):
    """A Swedish buyer's English title is English; without a statement or a
    detection the language is unknown, and is detected before translating."""
    assert source_language({"country": country}) is None
    assert source_language({"detail_country": country}) is None


def test_a_cohesion_title_takes_the_language_kohesio_states():
    assert source_language({"detail_country": "LTU", "title_lang": "en"}) == "en"
    assert "en" not in targets_for("en") and "lt" in targets_for("en")


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
@pytest.mark.asyncio
async def test_the_first_title_prices_the_rest(monkeypatch):
    """A round priced from the seed estimate would stake 32 titles on the
    guess; the probe measures one first."""
    rows = [_contract(f"n{i}", 1000 - i, f"title {i}") for i in range(10)]
    client = FakeClient(cost_per_item=0.10)                   # 285x the seed
    await _run(rows, client, monkeypatch, budget_usd=0.35)
    assert [len(items) for items, _t in client.calls] == [1, 2]
    assert sum(len(items) for items, _t in client.calls) * 0.10 <= 0.35


@pytest.mark.asyncio
async def test_it_stops_before_crossing_the_budget(monkeypatch, writes):
    rows = [_contract(f"n{i}", 1000 - i, f"title {i}") for i in range(10)]
    client = FakeClient(cost_per_item=0.10)
    progress = await _run(rows, client, monkeypatch, budget_usd=0.25)
    assert progress.spent_usd <= 0.25
    assert len(client.titles_sent) == 2
    assert "budget reached" in progress.stopped_because
    assert len(writes) == 2


@pytest.mark.asyncio
async def test_it_counts_what_was_charged(monkeypatch, writes):
    progress = await _run([_contract("n1", 100)], FakeClient(cost_per_item=0.00035),
                          monkeypatch)
    assert progress.spent_usd == pytest.approx(0.00035)
    assert progress.languages_written == len(EU_OFFICIAL_LANGS) - 1
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_nothing_already_translated_is_paid_for(monkeypatch, writes):
    rows = [_contract("done", 999, title_de="schon"), _contract("new", 5)]
    client = FakeClient()
    progress = await _run(rows, client, monkeypatch)
    assert client.titles_sent == ["Roboty budowlane"]
    assert progress.skipped_complete == 1
    assert [d.source_id for d in writes] == ["new"]


@pytest.mark.usefixtures("writes")
@pytest.mark.asyncio
async def test_it_works_richest_first(monkeypatch):
    """When only one title fits, it must be the biggest contract's."""
    rows = [_contract("big", 9_000_000, "Duży"), _contract("small", 12, "Mały")]
    client = FakeClient(cost_per_item=0.00035)
    await _run(rows, client, monkeypatch, budget_usd=0.0005)
    assert client.titles_sent == ["Duży"]


# ── batching ──────────────────────────────────────────────────


@pytest.mark.usefixtures("writes")
@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_a_duplicate_title_is_written_to_every_node(monkeypatch, writes):
    rows = [_contract("a", 9, "Same", "GBR"), _contract("b", 9, "Same", "GBR")]
    client = FakeClient()
    progress = await _run(rows, client, monkeypatch)
    assert client.titles_sent == ["Same"]
    assert sorted(d.source_id for d in writes) == ["a", "b"]
    assert progress.translated == 2 and progress.spent_usd == pytest.approx(0.00035)


@pytest.mark.asyncio
async def test_the_runner_never_writes_a_title_language(monkeypatch, writes):
    """title_lang belongs to the loader. A language the runner assumed (a
    country, "und") written there would read as a statement next time."""
    await _run([_contract("n", 9, "Anskaffelse", "NOR")], FakeClient(), monkeypatch)
    await _run([_contract("p", 9, "Roboty", "POL")], FakeClient(), monkeypatch)
    await _run([_contract("s", 9, "Travaux", "FRA", title_lang="fr")], FakeClient(), monkeypatch)
    assert [w.details["source_lang"] for w in writes] == [None, None, None]


# ── failure ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_failed_title_does_not_cost_the_others(monkeypatch, writes):
    rows = [_contract("a", 9, "good one"), _contract("b", 8, "bad one"),
            _contract("c", 7, "good two")]
    progress = await _run(rows, FakeClient(fail_titles={"bad one"}), monkeypatch)
    assert progress.failed == 1
    assert sorted(d.source_id for d in writes) == ["a", "c"]


@pytest.mark.asyncio
async def test_the_service_being_down_stops_the_run(monkeypatch, writes):
    progress = await _run([_contract("n", 9)], FakeClient(raises=LinguisticsUnavailable("503")),
                          monkeypatch)
    assert "unavailable" in progress.stopped_because and not writes


@pytest.mark.asyncio
async def test_a_hard_error_skips_the_round_not_the_run(monkeypatch, writes):
    progress = await _run([_contract("n", 9)], FakeClient(raises=LinguisticsError("400")),
                          monkeypatch)
    assert progress.failed == 1 and not writes


@pytest.mark.asyncio
async def test_a_cut_off_round_is_counted_at_its_worst(monkeypatch, writes):
    """A timeout can land after the provider billed the round; the report
    must not claim less than was spent."""
    class CutOffSecondRound(FakeClient):
        async def translate_batch_with_cost(self, items, targets):
            if self.calls:
                raise LinguisticsUnavailable("ReadTimeout")
            return await super().translate_batch_with_cost(items, targets)

    rows = [_contract(str(i), 100 - i, f"title {i}") for i in range(10)]
    progress = await _run(rows, CutOffSecondRound(cost_per_item=0.001), monkeypatch)
    # Probe of one title, then a round of the other nine that never came back.
    assert progress.spent_usd == pytest.approx(0.001)
    assert progress.unaccounted_usd == pytest.approx(9 * 0.001)
    assert len(writes) == 1 and "unavailable" in progress.stopped_because


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes")
async def test_the_client_waits_long_enough_for_a_round(monkeypatch):
    seen = {}

    def make_client(**kw):
        seen.update(kw)
        return FakeClient()

    monkeypatch.setattr(backfill, "LinguisticsClient", make_client)
    await run(FakeDriver([_contract("n", 9)]), "neo4j", label="Contract", limit=1,
              budget_usd=1.0, apply_changes=True)
    assert seen["timeout_s"] >= ROUND_TIMEOUT_S


@pytest.mark.asyncio
async def test_a_dry_run_sends_nothing_and_writes_nothing(monkeypatch, writes):
    client = FakeClient()
    progress = await _run([_contract("a", 9, "one"), _contract("b", 8, "two")], client,
                          monkeypatch, apply_changes=False)
    assert not client.calls and not writes and progress.spent_usd == 0
    assert progress.translated == 2
    assert progress.estimated_usd == pytest.approx(2 * SEED_COST_USD)


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes")
async def test_a_dry_run_says_where_the_budget_would_stop_it(monkeypatch):
    rows = [_contract(str(i), 100 - i, f"title {i}") for i in range(5)]
    progress = await _run(rows, FakeClient(), monkeypatch, apply_changes=False,
                          budget_usd=2.5 * SEED_COST_USD)
    assert progress.translated == 2 and progress.last_value == 99
    assert "after 2 titles" in progress.stopped_because


def test_the_dry_run_report_estimates_and_the_real_one_spends():
    progress = Progress(considered=3, distinct_titles=2, translated=2, skipped_complete=1,
                        spent_usd=0.0007, estimated_usd=0.0009, languages_written=46,
                        last_value=2_500_000_000)
    dry, real = progress.report("Contract", dry_run=True), progress.report("Contract", False)
    assert "would translate 2 of 3" in dry and "not reprocessed" in dry
    assert "$0.0009 (nothing sent)" in dry and "spent" not in dry
    assert "spent             : $0.0007" in real and "2,500,000,000 EUR" in real


def test_the_report_owns_up_to_spend_it_could_not_see():
    text = Progress(spent_usd=0.001, unaccounted_usd=0.009).report("Contract", dry_run=False)
    assert "up to $0.0090" in text


# ── source-language authority, and redoing wrong-source translations ──


@pytest.mark.parametrize("props, expected", [
    ({"title_lang": "en", "title_lang_detected": "fr"}, "en"),   # the notice beats a detection
    ({"title_lang_detected": "fr"}, "fr"),                        # a kept detection
    ({"title_lang": "no"}, UNDETERMINED),                         # outside the 24
    ({"title_lang_detected": "sr"}, UNDETERMINED),                # outside the 24
    ({"country": "FRA"}, None),                                   # unknown: detect first
])
def test_the_source_language_comes_from_the_best_authority(props, expected):
    assert source_language(props) == expected


def test_a_title_in_its_own_language_was_translated_from_another():
    assert translated_from_the_wrong_language({"title_lang": "en", "title_en": "copy"})
    assert translated_from_the_wrong_language({"title_lang_detected": "de", "title_de": "copy"})
    assert not translated_from_the_wrong_language({"title_lang": "fr", "title_en": "Works"})
    # unknown source: undecidable, never redone
    assert not translated_from_the_wrong_language({"country": "CHE", "title_de": "x"})


def test_a_wrong_source_translation_is_left_alone_unless_redo_is_asked():
    rows = [_contract("n", 9, "Marché de travaux", "FRA",
                      title_lang="en", title_en="Marché de travaux", title_de="x")]
    work, skipped = plan(rows, "Contract")
    assert not work and skipped == 1
    work, skipped = plan(rows, "Contract", redo=True)
    assert skipped == 0 and work[0].source_lang == "en" and work[0].clear == ("title_en",)


@pytest.mark.asyncio
async def test_a_redo_writes_the_new_translations_and_clears_the_stale_one(monkeypatch, writes):
    cleared = []

    async def fake_clear(_dest, node_ids, keys):
        cleared.append((list(node_ids), list(keys)))

    monkeypatch.setattr(backfill, "clear_properties", fake_clear)
    rows = [_contract("n", 9, "Construction works", "BEL",
                      title_lang="en", title_en="Construction works")]
    progress = await _run(rows, FakeClient(), monkeypatch, redo=True)
    assert "en" not in writes[0].details["translations"]
    assert cleared == [(["n"], ["title_en"])] and progress.redone == 1


@pytest.mark.asyncio
async def test_only_eu_title_keys_can_be_cleared():
    with pytest.raises(ValueError):
        await backfill.clear_properties(
            backfill.Destination(None, "neo4j", "Contract", True), ["n"], ["title_x; DETACH"])


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes")
async def test_a_dry_run_says_how_many_it_would_redo(monkeypatch):
    rows = [_contract("n", 9, "Construction works", "BEL",
                      title_lang="en", title_en="Construction works"),
            _contract("m", 8, "Roboty", "POL")]
    progress = await _run(rows, FakeClient(), monkeypatch, apply_changes=False, redo=True)
    assert progress.redone == 1 and progress.translated == 2


# ── detecting what nobody stated ──────────────────────────────


async def _run_on(driver, client, monkeypatch, **kw):
    monkeypatch.setattr(backfill, "LinguisticsClient", lambda **_kw: client)
    return await run(driver, "neo4j", label=kw.pop("label", "Contract"),
                     limit=len(driver.rows), budget_usd=kw.pop("budget_usd", 1.0),
                     apply_changes=kw.pop("apply_changes", True), **kw)


def _detections_kept(driver):
    return [(row["id"], row["lang"], params["model"])
            for query, params in driver.writes if "title_lang_detected" in query
            for row in params["rows"]]


@pytest.mark.asyncio
async def test_a_title_nobody_stated_is_detected_once_kept_and_translated_from(
        monkeypatch, writes):
    rows = [_contract("a", 9, "Travaux de voirie", title_lang=None),
            _contract("b", 8, "Travaux de voirie", title_lang=None),
            _contract("c", 7, "Roboty", title_lang="pl")]
    driver, client = FakeDriver(rows), FakeClient(detections={"Travaux de voirie": "fr"})
    progress = await _run_on(driver, client, monkeypatch)
    assert client.detect_calls == [["Travaux de voirie"]]          # asked once
    assert _detections_kept(driver) == [("a", "fr", "nebius:google/gemma-3-27b-it"),
                                        ("b", "fr", "nebius:google/gemma-3-27b-it")]
    langs = {t: lang for items, _t in client.calls for t, lang in items}
    assert langs == {"Travaux de voirie": "fr", "Roboty": "pl"}
    assert progress.detected == 2 and progress.detect_usd > 0
    assert sorted(w.source_id for w in writes) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_detection_is_kept_beside_title_lang_never_in_it(monkeypatch, writes):
    rows = [_contract("a", 9, "Travaux", title_lang=None)]
    driver = FakeDriver(rows)
    await _run_on(driver, FakeClient(detections={"Travaux": "fr"}), monkeypatch)
    detect_writes = [q for q, _p in driver.writes if "title_lang_detected" in q]
    assert detect_writes and all("n.title_lang =" not in q for q in detect_writes)
    assert writes[0].details["source_lang"] is None


@pytest.mark.asyncio
async def test_a_kept_detection_is_not_paid_for_again(monkeypatch, writes):
    rows = [_contract("a", 9, "Travaux", title_lang=None, title_lang_detected="fr")]
    client = FakeClient()
    await _run_on(FakeDriver(rows), client, monkeypatch)
    assert not client.detect_calls
    assert client.calls[0][0] == [("Travaux", "fr")] and len(writes) == 1


@pytest.mark.asyncio
async def test_a_title_with_no_words_is_kept_as_such_and_not_translated(monkeypatch, writes):
    rows = [_contract("a", 9, "209G001799 — 60338950", title_lang=None)]
    driver, client = FakeDriver(rows), FakeClient(detections={"209G001799 — 60338950": "und"})
    progress = await _run_on(driver, client, monkeypatch)
    assert _detections_kept(driver) == [("a", "und", "nebius:google/gemma-3-27b-it")]
    assert not client.calls and not writes and progress.no_words == 1
    assert has_no_words({"title_lang_detected": "und"})
    assert not has_no_words({"title_lang": "fr", "title_lang_detected": "und"})


@pytest.mark.asyncio
async def test_an_unanswered_title_waits_for_the_next_run(monkeypatch, writes):
    rows = [_contract("a", 9, "Mystery", title_lang=None), _contract("b", 8, "Roboty")]
    driver, client = FakeDriver(rows), FakeClient(detections={"Mystery": None})
    progress = await _run_on(driver, client, monkeypatch)
    assert _detections_kept(driver) == []
    assert [w.source_id for w in writes] == ["b"] and progress.undetected == 1
    assert "left for the next run" in progress.report("Contract", dry_run=False)


@pytest.mark.asyncio
async def test_detection_failing_stops_the_run_and_fails_it(monkeypatch, writes):
    rows = [_contract("a", 9, "Travaux", title_lang=None), _contract("b", 8, "Roboty")]
    client = FakeClient(detect_raises=LinguisticsError("status=404"))
    progress = await _run_on(FakeDriver(rows), client, monkeypatch)
    assert not client.calls and not writes
    assert progress.failures and "detection failed" in progress.stopped_because


@pytest.mark.asyncio
async def test_a_dry_run_detects_nothing_and_says_how_many_it_would(monkeypatch, writes):
    rows = [_contract("a", 9, "Travaux", title_lang=None), _contract("b", 8, "Roboty")]
    driver, client = FakeDriver(rows), FakeClient()
    progress = await _run_on(driver, client, monkeypatch, apply_changes=False)
    assert not client.detect_calls and not client.calls and not driver.writes and not writes
    assert progress.to_detect == 1 and progress.translated == 2
    assert "would detect      : 1" in progress.report("Contract", dry_run=True)


@pytest.mark.asyncio
async def test_a_redo_detects_the_translated_titles_whose_language_nobody_stated(
        monkeypatch, writes):
    """The identify-it prompt once copied titles into every language. With
    the language detected, such a copy is decidable, and redone."""
    rows = [_contract("a", 9, "Construction works", title_lang=None,
                      title_en="Construction works", title_fr="Construction works")]
    client = FakeClient(detections={"Construction works": "en"})
    progress = await _run_on(FakeDriver(rows), client, monkeypatch, redo=True)
    assert client.detect_calls == [["Construction works"]]
    assert client.calls[0][0] == [("Construction works", "en")]
    assert progress.redone == 1 and "en" not in writes[0].details["translations"]


# ── never twice ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.usefixtures("writes")
async def test_translations_record_the_title_they_were_made_from(monkeypatch):
    driver = FakeDriver([_contract("a", 9, "Roboty")])
    await _run_on(driver, FakeClient(), monkeypatch)
    marks = [p for q, p in driver.writes if "title_translated_from" in q]
    assert marks == [{"ids": ["a"], "title": "Roboty"}]


def test_a_retitled_node_is_translated_again_and_nothing_else_is():
    rows = [
        _contract("changed", 9, "Roboty drogowe", title_de="Bauarbeiten",
                  title_pl="Roboty budowlane", title_translated_from="Roboty budowlane"),
        _contract("same", 8, "Roboty", title_de="Arbeiten", title_translated_from="Roboty"),
        _contract("legacy", 7, "Dostawa", title_de="Lieferung"),      # made before the marker
    ]
    work, skipped = plan(rows, "Contract")
    assert [w.node_ids for w in work] == [["changed"]] and skipped == 2
    assert work[0].clear == ("title_pl",)     # the stale copy in its own language goes


# ── a euro cut-off instead of a percentage ────────────────────


@pytest.mark.asyncio
async def test_a_euro_cut_off_selects_everything_at_or_above_it_richest_first():
    driver = FakeDriver([_contract("n", 9)])
    await backfill.select_by_value(driver, "neo4j", "Contract", min_value=250e6)
    (query,) = driver.queries
    assert "n.value_eur >= $min_value" in query and "LIMIT" not in query
    assert query.endswith("ORDER BY n.value_eur DESC")


@pytest.mark.asyncio
async def test_a_grant_cut_off_is_on_the_eu_contribution():
    driver = FakeDriver([])
    await backfill.select_by_value(driver, "neo4j", "CohesionProject", min_value=70e6)
    assert "n.detail_eu_contribution >= $min_value" in driver.queries[0]


@pytest.mark.asyncio
async def test_a_cut_off_run_does_not_count_the_population(monkeypatch):
    """Counting 4.4M contracts is a full scan the cut-off does not need."""
    seen = {}

    async def fake_run(_driver, _database, **kw):
        seen.update(kw)
        return Progress()

    async def no_count(*_a):
        raise AssertionError("population counted")

    async def fake_driver():
        return FakeDriver([])

    async def no_close():
        return None

    monkeypatch.setattr(backfill, "run", fake_run)
    monkeypatch.setattr(backfill, "count_candidates", no_count)
    monkeypatch.setattr(backfill, "get_driver", fake_driver)
    monkeypatch.setattr(backfill, "close_driver", no_close)
    args = backfill.build_parser().parse_args(
        ["--label", "Contract", "--min-value-eur", "250e6"])
    assert await backfill.main_async(args) == 0
    assert seen["min_value"] == 250e6 and seen["limit"] is None
