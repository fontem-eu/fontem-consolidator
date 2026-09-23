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
from src.consolidator.rules.multilingual_shared import source_lang_from_country

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
    translated: int = 0
    skipped_complete: int = 0
    failed: int = 0
    spent_usd: float = 0.0
    languages_written: int = 0
    stopped_because: str = "finished the selection"
    last_value: float | None = None
    failures: list[str] = field(default_factory=list)

    def report(self, label: str, dry_run: bool) -> str:
        head = "would translate" if dry_run else "translated"
        lines = [
            f"{label}: {head} {self.translated} of {self.considered} considered",
            f"  languages written : {self.languages_written}",
            f"  already complete  : {self.skipped_complete}",
            f"  failed            : {self.failed}",
            f"  spent             : ${self.spent_usd:.4f}",
            f"  stopped because   : {self.stopped_because}",
        ]
        if self.last_value is not None:
            lines.append(f"  reached down to   : {self.last_value:,.0f} EUR")
        for f in self.failures[:5]:
            lines.append(f"  ! {f}")
        return "\n".join(lines)


def missing_targets(props: dict, source_lang: str) -> list[str]:
    """EU locales with no title_<lang> yet. Never the source language."""
    return [
        code for code in EU_OFFICIAL_LANGS
        if code != source_lang and not props.get(f"title_{code}")
    ]


async def select_by_value(
    driver: AsyncDriver, database: str, label: str, limit: int,
) -> list[dict]:
    """The `limit` most valuable titled nodes of `label`, richest first.

    Reads the whole property bag rather than a projection: deciding what is
    missing needs the 23 `title_<lang>` properties, and asking for them by
    name would be a 23-term RETURN that drifts the moment the language list
    changes.
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


async def _translate_one(
    client: LinguisticsClient, props: dict, label: str,
) -> tuple[dict[str, str], str, float]:
    """``(translations, source_lang, cost)`` for one node."""
    source_lang = source_lang_from_country(props.get(COUNTRY_PROPERTY[label]))
    targets = missing_targets(props, source_lang)
    if not targets:
        return {}, source_lang, 0.0
    translations, cost = await client.translate_with_cost(
        text=props["title"], source_lang=source_lang, targets=targets,
    )
    return translations, source_lang, cost


async def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    driver: AsyncDriver,
    database: str,
    *,
    label: str,
    limit: int,
    budget_usd: float,
    apply_changes: bool,
    backend: str = "nebius",
) -> Progress:
    """Translate down the value order until the selection or the budget ends."""
    progress = Progress()
    rows = await select_by_value(driver, database, label, limit)
    value_prop = VALUE_PROPERTY[label]
    billed_calls = 0

    async with LinguisticsClient(
        base_url=settings.linguistics_url,
        timeout_s=settings.linguistics_timeout_s,
        translation_backend=backend,
        embedding_backend=settings.linguistics_embedding_backend,
    ) as client:
        for props in rows:
            # Stop BEFORE the call that would cross the line, priced at what
            # calls have actually been costing. Stopping once spent exceeds
            # the budget would mean the last call had already crossed it —
            # a cap you notice rather than one you hold.
            expected = (progress.spent_usd / billed_calls) if billed_calls else SEED_COST_USD
            if progress.spent_usd + expected > budget_usd:
                progress.stopped_because = (
                    f"budget reached (${budget_usd:.2f}; next call ~${expected:.5f})"
                )
                break

            progress.considered += 1
            progress.last_value = props.get(value_prop)

            try:
                translations, source_lang, cost = await _translate_one(client, props, label)
            except (LinguisticsUnavailable, LinguisticsError) as exc:
                progress.failed += 1
                progress.failures.append(f"{props.get(ID_PROPERTY[label])}: {exc}")
                if isinstance(exc, LinguisticsUnavailable):
                    progress.stopped_because = f"linguistics unavailable: {exc}"
                    break
                continue

            progress.spent_usd += cost
            if cost:
                billed_calls += 1
            if not translations:
                progress.skipped_complete += 1
                continue

            progress.translated += 1
            progress.languages_written += len(translations)

            if apply_changes:
                await _enrich(driver, database, decision=Decision(
                    rule_name="backfill_translations",
                    action="enrich",
                    source_id=str(props.get(ID_PROPERTY[label])),
                    target_id=str(props.get(ID_PROPERTY[label])),
                    confidence=1.0,
                    entity_type=label,
                    details={
                        "field": "title",
                        "translations": translations,
                        "source_lang": source_lang,
                    },
                ))

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
        help="Write the translations. Without it, nothing is written and "
             "nothing is spent beyond the reads.",
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
