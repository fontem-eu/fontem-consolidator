"""TranslationEnrichmentCohesionProject — fills title_<lang> on :CohesionProject.

The third of a set: Authority gets `name_<lang>`, Contract and CohesionProject
get `title_<lang>`. Cohesion projects are how EU funds are actually assigned —
261,954 of them in prod, every one titled, none translated — and a Portuguese
reader cannot currently tell what a Polish-titled project funded.

Source language is what the loader states (English when the title came from
Kohesio's English-name column), else a detection the backfill runner kept.
Live, only grants at or above settings.live_translation_min_grant_eur.

Same v1 scope as contracts: title only. The description
(`detail_description`) is long-form and would multiply the token cost by an
order of magnitude for text nobody browses.
"""
from __future__ import annotations

from loguru import logger

from src.config import settings
from src.consolidator.clients.linguistics import (
    LinguisticsClient,
    LinguisticsError,
    LinguisticsUnavailable,
)
from src.consolidator.rules import title_translation
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


UNDETERMINED = title_translation.UNDETERMINED


def infer_source_lang(entity: Entity) -> str | None:
    """The title's language as Kohesio states it ("en" when the title came
    from its English-name column), else a detection the backfill runner
    kept; None when neither. Neither the country nor a blanket "English" is
    a statement: the country labelled a Lithuanian project's English title
    Lithuanian (2026-09-24), and assuming English would mislabel every
    programme-language fallback title."""
    return title_translation.source_language(entity.properties)


def missing_targets(entity: Entity) -> list[str]:
    """EU locales to translate into (see title_translation.missing_targets)."""
    return title_translation.missing_targets(entity.properties)


class TranslationEnrichmentCohesionProject(Rule):
    name = "translation_enrichment_cohesion_project"
    description = (
        "Fill missing EU-language translations (title_<lang>) on "
        ":CohesionProject by calling fontem-linguistics. Runs per-entity; "
        "never merges or flags. v1: title only."
    )
    entity_types = {"CohesionProject"}
    confidence = 1.0
    action = "enrich"

    async def applies(self, entity: Entity) -> bool:
        if not settings.linguistics_enabled:
            return False
        if not entity.properties.get("title"):
            return False
        if not title_translation.at_least(
                entity.properties, "detail_eu_contribution",
                settings.live_translation_min_grant_eur):
            return False
        return bool(missing_targets(entity))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        return [Candidate(entity=entity, context={"enrichment": True})]

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        title = entity.properties["title"]
        src_lang = infer_source_lang(entity)
        targets = missing_targets(entity)
        translations: dict[str, str] = {}

        backend = (
            candidate.context.get("translation_backend_override")
            or settings.linguistics_translation_backend
        )

        try:
            async with LinguisticsClient(
                base_url=settings.linguistics_url,
                timeout_s=settings.linguistics_timeout_s,
                translation_backend=backend,
                embedding_backend=settings.linguistics_embedding_backend,
            ) as client:
                if targets:
                    translations = await client.translate(
                        text=title, source_lang=src_lang, targets=targets,
                    )
        except LinguisticsUnavailable as exc:
            logger.warning(
                "translation_enrichment_cohesion_project: linguistics "
                "unavailable for {id}: {exc}", id=entity.id, exc=exc,
            )
            return self._noop(entity, "linguistics_unavailable")
        except LinguisticsError as exc:
            logger.error(
                "translation_enrichment_cohesion_project: linguistics hard "
                "error for {id}: {exc}", id=entity.id, exc=exc,
            )
            return self._noop(entity, "linguistics_error", message=str(exc))

        if not translations:
            return self._noop(entity, "already_complete")

        return Decision(
            rule_name=self.name,
            action="enrich",
            source_id=entity.id,
            target_id=entity.id,
            confidence=1.0,
            entity_type="CohesionProject",
            details={
                "field": "title",
                "translations": translations,
                # title_lang belongs to the loader, which states it from the
                # source. A translator writing its assumption there would
                # turn a guess into a statement for the next reader.
                "source_lang": None,
                "translated_from": title,
            },
        )

    def _noop(self, entity: Entity, reason: str, **extra: str) -> Decision:
        return Decision(
            rule_name=self.name, action="noop",
            source_id=entity.id, target_id=entity.id, confidence=0.0,
            entity_type="CohesionProject",
            details={"reason": reason, **extra},
        )
