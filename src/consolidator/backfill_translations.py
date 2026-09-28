"""Translate the titles that matter most, until the money runs out.

2.76 million contracts carry a title and 17,748 of them carry a French one.
Translating all of them into 23 languages is not a budget question anyone
wants to answer, so this does the opposite of a sweep: it spends a fixed
amount of money on the titles where a translation is worth the most, biggest
first, and stops.

Order of work
-------------
:Contract by ``value_eur`` descending, :CohesionProject by
``detail_eu_contribution`` descending. A EUR 2.5bn framework agreement read by
someone in another member state is the case this exists for; a EUR 900 office
supply order is not. Only canonically keyed contracts: one keyed by its legacy
OJ S reference is the duplicate of one keyed by its publication number
(gitops data-backlog Part 6), queued to be merged into it, and often carries a
value the quarantine has since withheld from its twin.

Which language
--------------
The notice's own statement (``title_lang``), else a language detected earlier
(``title_lang_detected``), else one detected now: linguistics asks the model,
and the answer is kept on the node with the model that gave it, so it is paid
for once. A title with no words to judge by is not translated; one the model
gave no answer for waits for the next run. Nothing is guessed from the
buyer's country, and nothing is written to ``title_lang``, which belongs to
the loader.

Never twice
-----------
A contract, not a notice, is what gets translated: a modification is another
notice of the same contract. Identical titles are translated once and written
to every node that carries them. A translated node is not reprocessed unless
its title has changed since (``title_translated_from``).

The budget is real
------------------
Every call returns what the provider charged (``cost_usd``, added in
fontem-linguistics#39) and the runner adds it up. It stops when the next call
could cross the cap — not when an estimate says it might have. The service
keeps its own daily cap underneath as a backstop, so a bug here cannot spend
more than that either.

Measured 2026-09-23 on google/gemma-3-27b-it: one 91-character title into 23
languages costs 138 prompt + 830 completion tokens, about $0.00035. The top
0.1% of titled contracts (~2,765) is therefore around $1.

Usage::

    python -m src.consolidator.backfill_translations --label Contract --top-percent 0.1
    python -m src.consolidator.backfill_translations --label Contract --min-value-eur 250e6

The 0.1% cut-offs, measured on prod 2026-09-28 among titled canonical nodes
with a value and rounded: contracts at EUR 250M (1,959 of 2,015,174,
0.097%), cohesion grants at EUR 70M of EU contribution (250 of 245,262,
0.102%). A euro figure says what the selection is in terms a reader can
check; a percentage moves with every load.
    python -m src.consolidator.backfill_translations --label Contract --budget-usd 5.40 --apply
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field

from loguru import logger
from neo4j import AsyncDriver

from src.config import settings
from src.consolidator.actions import _enrich  # noqa: PLC2701  (one write path)
from src.consolidator.neo4j.client import close_driver, get_driver
from src.consolidator.clients.linguistics import (
    EU_OFFICIAL_LANGS,
    LinguisticsClient,
    LinguisticsError,
    LinguisticsUnavailable,
)
from src.consolidator.rules.base import Decision

#: What one title into 23 languages costs, measured 2026-09-23 on
#: google/gemma-3-27b-it (138 prompt + 830 completion tokens at
#: $0.13/$0.40 per Mtok). Used only to decide whether there is room for the
#: FIRST call; after that the runner uses what it has actually been charged.
SEED_COST_USD = 0.00035

#: What each label is worth, and which property says so. Ordering by anything
#: else would defeat the point of the exercise.
VALUE_PROPERTY: dict[str, str] = {
    "Contract": "value_eur",
    "CohesionProject": "detail_eu_contribution",
}
#: Which nodes of each label are the canonical record. A contract keyed by
#: its legacy OJ S reference ("2021/S 129-344226") has, or will have, a twin
#: keyed by its publication number ("344226-2021"), which is the one kept.
#: A substring test rather than the anchored regex: every legacy key has
#: "/S " and no canonical one (a publication number or an eForms UUID) can,
#: and over 4.4M contracts the regex took a count past Neo4j's 90 s limit.
CANONICAL_FILTER: dict[str, str] = {
    "Contract": "AND NOT n.ted_notice_id CONTAINS '/S '",
    "CohesionProject": "",
}
ID_PROPERTY: dict[str, str] = {
    "Contract": "ted_notice_id",
    "CohesionProject": "disclosure_id",
}


# A report, not behaviour: each field is a line someone paying for the run
# would ask about. Grouping them into sub-structs would only make the report
# harder to read than the thing it reports on.
@dataclass
class Progress:  # pylint: disable=too-many-instance-attributes
    """What the run did, in the terms someone paying for it would ask."""

    considered: int = 0
    distinct_titles: int = 0
    translated: int = 0
    billed_titles: int = 0
    max_title_cost: float = 0.0
    skipped_complete: int = 0
    failed: int = 0
    spent_usd: float = 0.0
    unaccounted_usd: float = 0.0
    estimated_usd: float = 0.0
    languages_written: int = 0
    redone: int = 0
    detected: int = 0
    to_detect: int = 0
    undetected: int = 0
    no_words: int = 0
    retitled: int = 0
    detect_usd: float = 0.0
    stopped_because: str = "finished the selection"
    last_value: float | None = None
    failures: list[str] = field(default_factory=list)

    def report(self, label: str, dry_run: bool) -> str:
        head = "would translate" if dry_run else "translated"
        money = (f"  estimated cost    : ${self.estimated_usd:.4f} (nothing sent)" if dry_run
                 else f"  spent             : ${self.spent_usd:.4f}")
        lines = [
            f"{label}: {head} {self.translated} of {self.considered} considered",
            f"  distinct titles   : {self.distinct_titles}",
            f"  languages written : {self.languages_written}",
            f"  already translated: {self.skipped_complete} (skipped, not reprocessed)",
            f"  wrong source, redone: {self.redone}",
            f"  title changed, redone: {self.retitled}",
            f"  language detected : {self.detected}"
            + (f" (${self.detect_usd:.4f})" if self.detect_usd else ""),
            f"  no words to translate: {self.no_words}",
            f"  failed            : {self.failed}",
            money,
        ]
        if dry_run and self.to_detect:
            lines.insert(-1, f"  would detect      : {self.to_detect} (no stated language)")
        if self.undetected:
            lines.insert(-1, f"  language unknown  : {self.undetected} (left for the next run)")
        if self.unaccounted_usd:
            lines.append(f"  unaccounted       : up to ${self.unaccounted_usd:.4f}"
                         " (a round was cut off after it was sent)")
        lines.append(f"  stopped because   : {self.stopped_because}")
        if self.last_value is not None:
            lines.append(f"  reached down to   : {self.last_value:,.0f} EUR")
        for f in self.failures[:5]:
            lines.append(f"  ! {f}")
        return "\n".join(lines)


async def select_by_value(
    driver: AsyncDriver, database: str, label: str, limit: int | None = None,
    *, min_value: float | None = None,
) -> list[dict]:
    """The titled nodes of `label` worth at least `min_value` euros, or else
    the `limit` most valuable ones, richest first.

    Reads the whole property bag rather than a projection: deciding whether
    a node is already translated needs its `title_<lang>` properties, and
    naming 24 of them in a RETURN would drift when the language list does.
    No index on the value property in prod, so this is a label scan — about a
    minute on 2.8M contracts. Run once per backfill, not per round.
    """
    value_prop = VALUE_PROPERTY[label]
    worth = ">= $min_value" if min_value is not None else "IS NOT NULL"
    query = (
        f"MATCH (n:{label}) "
        f"WHERE n.title IS NOT NULL AND n.{value_prop} {worth} {CANONICAL_FILTER[label]} "
        f"RETURN properties(n) AS props ORDER BY n.{value_prop} DESC"
        + (" LIMIT $limit" if min_value is None else "")
    )
    async with driver.session(database=database) as session:
        result = await session.run(query, limit=limit, min_value=min_value)
        return [record["props"] async for record in result]


async def count_candidates(driver: AsyncDriver, database: str, label: str) -> int:
    """Titled canonical nodes with a value — the population the percentage is of."""
    value_prop = VALUE_PROPERTY[label]
    query = (
        f"MATCH (n:{label}) "
        f"WHERE n.title IS NOT NULL AND n.{value_prop} IS NOT NULL {CANONICAL_FILTER[label]} "
        "RETURN count(n) AS n"
    )
    async with driver.session(database=database) as session:
        result = await session.run(query)
        record = await result.single()
        return int(record["n"]) if record else 0


#: BCP-47 "undetermined": linguistics asks the model to identify the
#: language rather than being told one.
UNDETERMINED = "und"

#: Distinct titles per round. Small enough to keep the order close to strict
#: value order under a budget cut-off, large enough to be worth a batch call.
ROUND_SIZE = 32

#: How long to wait for one round. The service translates a batch through an
#: 8-wide window, so 32 titles are four provider calls back to back, each
#: allowed 120 s plus retries there. The consolidator's usual 60 s is sized
#: for one title and would cut most rounds off after they had been paid for.
ROUND_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class Destination:
    """Where translated titles are written — or, in a dry run, are not."""

    driver: AsyncDriver
    database: str
    label: str
    apply_changes: bool


@dataclass
class WorkItem:
    """One distinct title to translate, and every node that carries it."""

    title: str
    source_lang: str
    targets: list[str]
    node_ids: list[str]
    value: float
    #: title_<lang> keys a redo removes: the source language's, which the
    #: wrong run wrote and a correct translation never targets.
    clear: tuple[str, ...] = ()


def already_translated(props: dict) -> bool:
    """Translated, and from the title it carries now. The owner's rule is not
    to reprocess, so one existing language counts; a node translated before
    the source title was recorded is taken to be current."""
    if not any(props.get(f"title_{code}") for code in EU_OFFICIAL_LANGS):
        return False
    return not title_changed(props)


def title_changed(props: dict) -> bool:
    """The title differs from the one its translations were made from."""
    source = props.get("title_translated_from")
    return source is not None and source != props.get("title")


#: What detection answers for a title with no words to judge by: a
#: reference number, a code, a bare name. Nothing to translate.
NO_WORDS = UNDETERMINED


def source_language(props: dict) -> str | None:
    """The title's language, from the best authority available.

    The source's own statement (``title_lang``: the TED notice's language,
    Kohesio's English-name column), else a language detected from the title
    and kept on the node (``title_lang_detected``). None when neither is
    known: detect it before translating. A language outside the 24 has no
    name linguistics can put in a prompt, so the model identifies it ("und").
    """
    for key in ("title_lang", "title_lang_detected"):
        code = (props.get(key) or "").lower()
        if code:
            return code if code in EU_OFFICIAL_LANGS else UNDETERMINED
    return None


def has_no_words(props: dict) -> bool:
    """Detection found nothing to translate, and no source says otherwise."""
    return (not props.get("title_lang")
            and (props.get("title_lang_detected") or "").lower() == NO_WORDS)


def translated_from_the_wrong_language(props: dict) -> bool:
    """A title translated into its own language was translated from another.

    Translation never targets its source, so a ``title_<source>`` on the node
    means the run that wrote it assumed a different source: a country guess
    the notice contradicts, Kohesio's English read as the country's language,
    or the identify-it prompt that copied the title into every language.
    Only decidable when the source is known.
    """
    src = source_language(props)
    return src not in (None, UNDETERMINED) and bool(props.get(f"title_{src}"))


def targets_for(source_lang: str) -> list[str]:
    """Every EU language except the source; all of them when it is unknown,
    since the model returns the source's own entry unchanged."""
    if source_lang == UNDETERMINED:
        return list(EU_OFFICIAL_LANGS)
    return [code for code in EU_OFFICIAL_LANGS if code != source_lang]


def needs_translation(props: dict, redo: bool) -> tuple[bool, bool]:
    """``(translate it, redo it)``: work that is new, retitled, or (with
    ``redo``) translated from the wrong language."""
    wrong = redo and translated_from_the_wrong_language(props)
    return (wrong or not already_translated(props)), wrong


def plan(
    rows: list[dict], label: str, *, redo: bool = False,
) -> tuple[list[WorkItem], int]:
    """Value-ordered, de-duplicated work, and how many nodes were skipped.

    Identical (title, language) pairs collapse into one item: framework
    agreements republish the same title, and translating it once is the same
    translation at a fraction of the cost. A node redone because it was
    translated from the wrong language, or because its title changed, has its
    stale source-language title cleared when the new translations land.
    Nodes whose language is still unknown are not planned; ``run`` detects
    it first.
    """
    by_key: dict[tuple[str, str], WorkItem] = {}
    skipped = 0
    for props in rows:
        todo, wrong = needs_translation(props, redo)
        lang = source_language(props)
        if not todo or lang is None or has_no_words(props):
            skipped += not todo
            continue
        key = (props["title"], lang)
        node_id = str(props.get(ID_PROPERTY[label]))
        stale = (wrong or title_changed(props)) and props.get(f"title_{lang}")
        clear = (f"title_{lang}",) if stale else ()
        if key in by_key:
            by_key[key].node_ids.append(node_id)
            by_key[key].clear = tuple(sorted(set(by_key[key].clear) | set(clear)))
            continue
        by_key[key] = WorkItem(
            title=props["title"], source_lang=lang, targets=targets_for(lang),
            node_ids=[node_id], value=float(props.get(VALUE_PROPERTY[label]) or 0),
            clear=clear,
        )
    return list(by_key.values()), skipped


def tally_unplanned(rows: list[dict], progress: "Progress", *, redo: bool) -> None:
    """Count the nodes plan() left out for want of a language, by reason."""
    for props in rows:
        todo = needs_translation(props, redo)[0]
        if todo and has_no_words(props):
            progress.no_words += 1
        elif needs_detection(props, redo):
            progress.undetected += 1
        elif todo and title_changed(props):
            progress.retitled += 1


def needs_detection(props: dict, redo: bool) -> bool:
    """No known language, and one is needed: to translate the title, or,
    in a redo, to tell whether its translations came from the wrong one."""
    if source_language(props) is not None or has_no_words(props):
        return False
    return redo or needs_translation(props, redo)[0]


def items_that_fit(progress: "Progress", wanted: int, budget_usd: float) -> int:
    """How many more titles the budget allows before the next round.

    Nothing has been billed yet: send ONE title and measure it. Pricing a
    whole round from the seed estimate would stake up to ROUND_SIZE titles on
    that guess being right — at a dearer model, a round could overshoot the
    budget by 30 titles before the runner learned the real price.

    After that, each title is priced at the most expensive one seen so far,
    not the average: titles vary in length, and the cap should hold for the
    long ones too.
    """
    room = budget_usd - progress.spent_usd
    if not progress.billed_titles:
        return 1 if room >= SEED_COST_USD else 0
    per_title = progress.max_title_cost
    return max(0, min(wanted, int(room // per_title))) if per_title > 0 else wanted


async def translate_round(
    client: LinguisticsClient, items: list[WorkItem],
) -> list[tuple[WorkItem, dict[str, str], float, str | None]]:
    """One batch call per source language in the round, results in item order."""
    by_lang: dict[str, list[WorkItem]] = {}
    for item in items:
        by_lang.setdefault(item.source_lang, []).append(item)

    done: dict[int, tuple[dict[str, str], float, str | None]] = {}
    for lang, group in by_lang.items():
        results = await client.translate_batch_with_cost(
            [(i.title, lang) for i in group], targets_for(lang),
        )
        for item, result in zip(group, results):
            done[id(item)] = result
    return [(item, *done[id(item)]) for item in items]


async def bank(
    dest: Destination,
    outcome: tuple[WorkItem, dict[str, str], float, str | None],
    progress: "Progress",
) -> None:
    """Record one title's result, and write it to every node that carries it."""
    item, translations, cost, error = outcome
    progress.spent_usd += cost
    if cost:
        progress.billed_titles += 1
        progress.max_title_cost = max(progress.max_title_cost, cost)
    if error or not translations:
        progress.failed += 1
        progress.failures.append(f"{item.node_ids[0]}: {error or 'empty response'}")
        return

    progress.translated += len(item.node_ids)
    progress.languages_written += len(translations) * len(item.node_ids)
    if not dest.apply_changes:
        return
    # title_lang belongs to the loader, which states it from the source. The
    # runner writes translations only: writing the language it assumed would
    # turn a country guess into a statement the next run believes.
    for node_id in item.node_ids:
        await _enrich(dest.driver, dest.database, decision=Decision(
            rule_name="backfill_translations", action="enrich",
            source_id=node_id, target_id=node_id, confidence=1.0,
            entity_type=dest.label,
            details={"field": "title", "translations": translations,
                     "source_lang": None},
        ))
    await set_translated_from(dest, item.node_ids, item.title)
    stale = [k for k in item.clear if k[len("title_"):] not in translations]
    if stale:
        progress.redone += len(item.node_ids)
        await clear_properties(dest, item.node_ids, stale)


async def set_translated_from(dest: Destination, node_ids: list[str], title: str) -> None:
    """Record the title the translations were made from, so a later change
    to it is seen as one rather than left under stale translations."""
    query = (f"MATCH (n:{dest.label}) WHERE n.{ID_PROPERTY[dest.label]} IN $ids "
             "SET n.title_translated_from = $title")
    async with dest.driver.session(database=dest.database) as session:
        await session.run(query, ids=node_ids, title=title)


#: Texts per /detect request, the service's limit.
DETECT_REQUEST_SIZE = 256


async def detect_languages(
    client: LinguisticsClient, dest: Destination, rows: list[dict],
    progress: "Progress", *, redo: bool,
) -> None:
    """Detect the language of every title that needs one, keep it on its
    nodes, and note it on the rows for plan().

    Each distinct title is asked once. "und" (no words) is kept too, so the
    next run does not pay to ask again; a title the model did not answer is
    left unknown and asked next time.
    """
    wanted: dict[str, list[dict]] = {}
    for props in rows:
        if needs_detection(props, redo):
            wanted.setdefault(props["title"], []).append(props)
    titles = list(wanted)
    for start in range(0, len(titles), DETECT_REQUEST_SIZE):
        chunk = titles[start:start + DETECT_REQUEST_SIZE]
        langs, model, cost = await client.detect(chunk)
        progress.spent_usd += cost
        progress.detect_usd += cost
        found = _note_detected(dict(zip(chunk, langs)), wanted, ID_PROPERTY[dest.label])
        progress.detected += len(found)
        await persist_detected(dest, found, model or "unknown")


def _note_detected(
    answers: dict[str, str | None], wanted: dict[str, list[dict]], id_prop: str,
) -> list[dict]:
    """Put each answered language on the rows carrying that title; return
    the ``{id, lang}`` rows to keep on the graph."""
    found = []
    for title, lang in answers.items():
        if not lang:
            continue
        for props in wanted[title]:
            props["title_lang_detected"] = lang
            found.append({"id": str(props.get(id_prop)), "lang": lang})
    return found


async def persist_detected(dest: Destination, found: list[dict], model: str) -> None:
    """Keep each detected language on its node, with where it came from.

    Beside ``title_lang``, never in it: the loader states that one from the
    source, and a detection written there would read as a statement.
    """
    if not found or not dest.apply_changes:
        return
    query = (f"UNWIND $rows AS row MATCH (n:{dest.label} "
             f"{{{ID_PROPERTY[dest.label]}: row.id}}) "
             "SET n.title_lang_detected = row.lang, n.title_lang_detected_by = $model, "
             "n.title_lang_detected_at = datetime()")
    async with dest.driver.session(database=dest.database) as session:
        await session.run(query, rows=found, model=model)


async def clear_properties(dest: Destination, node_ids: list[str], keys: list[str]) -> None:
    """Remove stale ``title_<lang>`` properties a redo supersedes.

    Keys are interpolated, so only ``title_`` plus one of the EU codes is
    accepted; anything else is a bug upstream, not input to trust.
    """
    allowed = {f"title_{code}" for code in EU_OFFICIAL_LANGS}
    if not set(keys) <= allowed:
        raise ValueError(f"refusing to clear {sorted(set(keys) - allowed)}")
    sets = ", ".join(f"n.{k} = null" for k in sorted(keys))
    query = (f"MATCH (n:{dest.label}) WHERE n.{ID_PROPERTY[dest.label]} IN $ids "
             f"SET {sets}")
    async with dest.driver.session(database=dest.database) as session:
        await session.run(query, ids=node_ids)


def estimate(work: list[WorkItem], progress: "Progress", budget_usd: float) -> None:
    """What a real run would do, priced at the measured per-title cost.

    Sends nothing. A dry run that called the provider would be a paid run
    whose results were thrown away.
    """
    affordable = work[:int(budget_usd // SEED_COST_USD)]
    progress.to_detect = progress.undetected
    progress.undetected = 0
    progress.translated = sum(len(i.node_ids) for i in affordable)
    progress.languages_written = sum(len(i.targets) * len(i.node_ids) for i in affordable)
    progress.estimated_usd = len(affordable) * SEED_COST_USD
    progress.redone = sum(len(i.node_ids) for i in affordable if i.clear)
    if affordable:
        progress.last_value = affordable[-1].value
    progress.stopped_because = (
        "dry run: the whole selection fits the budget" if len(affordable) == len(work)
        else f"dry run: the budget would stop it after {len(affordable)} titles"
    )


async def work_down(
    client: LinguisticsClient, work: list[WorkItem], progress: "Progress",
    dest: Destination, budget_usd: float,
) -> None:
    """Take rounds off the value-ordered work until it or the budget ends."""
    cursor = 0
    while cursor < len(work):
        fit = items_that_fit(progress, min(ROUND_SIZE, len(work) - cursor), budget_usd)
        if fit == 0:
            progress.stopped_because = f"budget reached (${budget_usd:.2f})"
            return
        round_items = work[cursor:cursor + fit]
        cursor += fit
        progress.last_value = round_items[-1].value
        try:
            outcomes = await translate_round(client, round_items)
        except LinguisticsUnavailable as exc:
            # A timeout may land after the provider has billed the round, and
            # the per-item costs went down with the response. Count it at its
            # worst so the report never claims less than was spent.
            progress.unaccounted_usd += fit * max(progress.max_title_cost, SEED_COST_USD)
            progress.stopped_because = f"linguistics unavailable: {exc}"
            return
        except LinguisticsError as exc:
            progress.failed += len(round_items)
            progress.failures.append(f"round at {cursor - fit}: {exc}")
            continue
        for outcome in outcomes:
            await bank(dest, outcome, progress)


async def run(  # pylint: disable=too-many-arguments
    driver: AsyncDriver,
    database: str,
    *,
    label: str,
    limit: int | None = None,
    budget_usd: float,
    apply_changes: bool,
    backend: str = "nebius",
    redo: bool = False,
    min_value: float | None = None,
) -> Progress:
    """Select the nodes worth at least `min_value` (else the richest
    `limit`), skip the translated, detect the languages nobody stated, then
    translate down the value order until the selection or the budget runs
    out."""
    rows = await select_by_value(driver, database, label, limit, min_value=min_value)
    dest = Destination(driver, database, label, apply_changes)
    progress = Progress(considered=len(rows))
    if not apply_changes:
        # A dry run sends nothing, detection included: the titles without a
        # language are priced as translations and counted as to be detected.
        work, progress.skipped_complete = plan(
            [p if source_language(p) or has_no_words(p) else {**p, "title_lang": "und"}
             for p in rows], label, redo=redo)
        tally_unplanned(rows, progress, redo=redo)
        progress.distinct_titles = len(work)
        estimate(work, progress, budget_usd)
        return progress
    async with LinguisticsClient(
        base_url=settings.linguistics_url,
        timeout_s=max(settings.linguistics_timeout_s, ROUND_TIMEOUT_S),
        translation_backend=backend,
        embedding_backend=settings.linguistics_embedding_backend,
    ) as client:
        try:
            await detect_languages(client, dest, rows, progress, redo=redo)
        except (LinguisticsUnavailable, LinguisticsError) as exc:
            progress.stopped_because = f"language detection failed: {exc}"
            progress.failures.append(f"detect: {exc}")
            return progress
        work, progress.skipped_complete = plan(rows, label, redo=redo)
        tally_unplanned(rows, progress, redo=redo)
        progress.distinct_titles = len(work)
        await work_down(client, work, progress, dest, budget_usd)
    return progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=sorted(VALUE_PROPERTY), default="Contract")
    parser.add_argument(
        "--top-percent", type=float, default=0.1,
        help="How much of the titled population to consider, richest first.",
    )
    parser.add_argument(
        "--min-value-eur", type=float, default=None,
        help="Consider every titled node worth at least this many euros "
             "(value_eur for contracts, detail_eu_contribution for cohesion "
             "projects), richest first. Replaces --top-percent.",
    )
    parser.add_argument(
        "--budget-usd", type=float, default=5.40,
        help="Hard ceiling on provider spend. EUR 5 at 1.08 USD/EUR.",
    )
    parser.add_argument(
        "--backend", default="nebius",
        help="Translation backend the linguistics service should use.",
    )
    parser.add_argument(
        "--redo-wrong-source", action="store_true",
        help="Also re-translate titles that were translated from the wrong "
             "language (they carry a title in their own language), clearing "
             "that stale one. Off by default: translated titles are not "
             "reprocessed.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Translate and write. Without it, nothing is sent to the provider "
             "and nothing is written: the selection is planned and priced at "
             "the measured per-title cost.",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    driver = await get_driver()
    database = settings.neo4j_database
    try:
        limit = None
        if args.min_value_eur is not None:
            logger.info("{label}: everything worth at least EUR {v:,.0f}, budget ${budget:.2f}",
                        label=args.label, v=args.min_value_eur, budget=args.budget_usd)
        else:
            total = await count_candidates(driver, database, args.label)
            limit = max(1, round(total * args.top_percent / 100))
            logger.info(
                "{label}: {total:,} titled with a value; top {pct}% = {limit:,} nodes, "
                "budget ${budget:.2f}",
                label=args.label, total=total, pct=args.top_percent,
                limit=limit, budget=args.budget_usd,
            )
        progress = await run(
            driver, database,
            label=args.label, limit=limit, budget_usd=args.budget_usd,
            apply_changes=args.apply, backend=args.backend,
            redo=args.redo_wrong_source, min_value=args.min_value_eur,
        )
        print(progress.report(args.label, dry_run=not args.apply))
        return 0 if not progress.failures else 1
    finally:
        await close_driver()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
