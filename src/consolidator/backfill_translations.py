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
supply order is not.

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
from src.consolidator.rules.multilingual_shared import (
    COUNTRY_PRIMARY_LANG,
    source_lang_from_country,
)

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
COUNTRY_PROPERTY: dict[str, str] = {
    "Contract": "country",
    "CohesionProject": "detail_country",
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
            f"  failed            : {self.failed}",
            money,
        ]
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
    driver: AsyncDriver, database: str, label: str, limit: int,
) -> list[dict]:
    """The `limit` most valuable titled nodes of `label`, richest first.

    Reads the whole property bag rather than a projection: deciding whether
    a node is already translated needs its `title_<lang>` properties, and
    naming 24 of them in a RETURN would drift when the language list does.
    No index on the value property in prod, so this is a label scan — about a
    minute on 2.8M contracts. Run once per backfill, not per round.
    """
    value_prop = VALUE_PROPERTY[label]
    query = (
        f"MATCH (n:{label}) "
        f"WHERE n.title IS NOT NULL AND n.{value_prop} IS NOT NULL "
        f"RETURN properties(n) AS props ORDER BY n.{value_prop} DESC LIMIT $limit"
    )
    async with driver.session(database=database) as session:
        result = await session.run(query, limit=limit)
        return [record["props"] async for record in result]


async def count_candidates(driver: AsyncDriver, database: str, label: str) -> int:
    """Titled nodes with a value — the population the percentage is of."""
    value_prop = VALUE_PROPERTY[label]
    query = (
        f"MATCH (n:{label}) "
        f"WHERE n.title IS NOT NULL AND n.{value_prop} IS NOT NULL "
        "RETURN count(n) AS n"
    )
    async with driver.session(database=database) as session:
        result = await session.run(query)
        record = await result.single()
        return int(record["n"]) if record else 0


#: BCP-47 "undetermined": linguistics asks the model to identify the
#: language rather than being told one.
UNDETERMINED = "und"

#: Countries whose notices are not reliably in the language the country map
#: gives. Belgium publishes in French and Dutch; Luxembourg in French, German
#: and English; most Maltese TED notices are in English; Finland publishes in
#: Swedish too. For these, and for any country the map does not know (Norway,
#: Switzerland, the candidate countries), the model identifies the language.
UNRELIABLE_COUNTRIES = frozenset({"BEL", "LUX", "MLT", "FIN"})

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


def already_translated(props: dict) -> bool:
    """Any translated title at all. The owner's rule is not to reprocess."""
    return any(props.get(f"title_{code}") for code in EU_OFFICIAL_LANGS)


#: Labels whose titles arrive in one language whatever the country. Kohesio
#: publishes every project title in English, its own rendering of the
#: beneficiary's original: a Lithuanian project's title is English, and
#: reading it as Lithuanian both mislabels it and never asks for Lithuanian.
FIXED_SOURCE_LANGUAGE: dict[str, str] = {"CohesionProject": "en"}


def source_language(props: dict, label: str) -> str:
    """The title's language when the source or the country tells us, else "und".

    Guessing wrong is not neutral: a Norwegian title labelled English is
    translated from the wrong language, and English itself is never
    requested because the runner believes it already has it.
    """
    if label in FIXED_SOURCE_LANGUAGE:
        return FIXED_SOURCE_LANGUAGE[label]
    country = (props.get(COUNTRY_PROPERTY[label]) or "").upper()
    if country in UNRELIABLE_COUNTRIES or country not in COUNTRY_PRIMARY_LANG:
        return UNDETERMINED
    return source_lang_from_country(country)


def targets_for(source_lang: str) -> list[str]:
    """Every EU language except the source; all of them when it is unknown,
    since the model returns the source's own entry unchanged."""
    if source_lang == UNDETERMINED:
        return list(EU_OFFICIAL_LANGS)
    return [code for code in EU_OFFICIAL_LANGS if code != source_lang]


def plan(rows: list[dict], label: str) -> tuple[list[WorkItem], int]:
    """Value-ordered, de-duplicated work, and how many nodes were skipped.

    Identical (title, language) pairs collapse into one item: framework
    agreements republish the same title, and translating it once is the same
    translation at a fraction of the cost.
    """
    by_key: dict[tuple[str, str], WorkItem] = {}
    skipped = 0
    for props in rows:
        if already_translated(props):
            skipped += 1
            continue
        lang = source_language(props, label)
        key = (props["title"], lang)
        node_id = str(props.get(ID_PROPERTY[label]))
        if key in by_key:
            by_key[key].node_ids.append(node_id)
            continue
        by_key[key] = WorkItem(
            title=props["title"], source_lang=lang, targets=targets_for(lang),
            node_ids=[node_id], value=float(props.get(VALUE_PROPERTY[label]) or 0),
        )
    return list(by_key.values()), skipped


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
    # "und" means we do not know the title's language, so we do not claim one.
    source = None if item.source_lang == UNDETERMINED else item.source_lang
    for node_id in item.node_ids:
        await _enrich(dest.driver, dest.database, decision=Decision(
            rule_name="backfill_translations", action="enrich",
            source_id=node_id, target_id=node_id, confidence=1.0,
            entity_type=dest.label,
            details={"field": "title", "translations": translations,
                     "source_lang": source},
        ))


def estimate(work: list[WorkItem], progress: "Progress", budget_usd: float) -> None:
    """What a real run would do, priced at the measured per-title cost.

    Sends nothing. A dry run that called the provider would be a paid run
    whose results were thrown away.
    """
    affordable = work[:int(budget_usd // SEED_COST_USD)]
    progress.translated = sum(len(i.node_ids) for i in affordable)
    progress.languages_written = sum(len(i.targets) * len(i.node_ids) for i in affordable)
    progress.estimated_usd = len(affordable) * SEED_COST_USD
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
    limit: int,
    budget_usd: float,
    apply_changes: bool,
    backend: str = "nebius",
) -> Progress:
    """Select the richest `limit` nodes, skip the translated, translate the
    rest down the value order until the selection or the budget runs out."""
    rows = await select_by_value(driver, database, label, limit)
    work, skipped = plan(rows, label)
    progress = Progress(considered=len(rows), skipped_complete=skipped,
                        distinct_titles=len(work))
    if not apply_changes:
        estimate(work, progress, budget_usd)
        return progress
    async with LinguisticsClient(
        base_url=settings.linguistics_url,
        timeout_s=max(settings.linguistics_timeout_s, ROUND_TIMEOUT_S),
        translation_backend=backend,
        embedding_backend=settings.linguistics_embedding_backend,
    ) as client:
        await work_down(client, work, progress,
                        Destination(driver, database, label, apply_changes), budget_usd)
    return progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=sorted(VALUE_PROPERTY), default="Contract")
    parser.add_argument(
        "--top-percent", type=float, default=0.1,
        help="How much of the titled population to consider, richest first.",
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
        )
        print(progress.report(args.label, dry_run=not args.apply))
        return 0 if not progress.failures else 1
    finally:
        await close_driver()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
