"""TranslationEnrichmentCohesionProject — fills title_<lang> on :CohesionProject.

The third of a set: Authority gets `name_<lang>`, Contract and CohesionProject
get `title_<lang>`. Cohesion projects are how EU funds are actually assigned —
261,954 of them in prod, every one titled, none translated — and a Portuguese
reader cannot currently tell what a Polish-titled project funded.

Source language comes from `detail_country`, since Kohesio carries no title
language of its own. Same v1 scope as contracts: title only. The description
(`detail_description`) is long-form and would multiply the token cost by an
order of magnitude for text nobody browses.
"""
from __future__ import annotations

from loguru import logger

from src.config import settings
from src.consolidator.clients.linguistics import (
    EU_OFFICIAL_LANGS,
    LinguisticsClient,
    LinguisticsError,
    LinguisticsUnavailable,
)
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule
from src.consolidator.rules.multilingual_shared import source_lang_from_country


def infer_source_lang(entity: Entity) -> str:
    """Best-effort source language for a project title.

    Kohesio has no title-language field, so the beneficiary country is the
    signal. Falls through to "en" for unknowns, which is what the shared
    helper does.
    """
    explicit = (entity.properties.get("title_lang") or "").lower()
    if explicit:
        return explicit
    return source_lang_from_country(entity.properties.get("detail_country"))


def missing_targets(entity: Entity) -> list[str]:
    """EU locales with no title_<lang> on the node yet."""
    src = infer_source_lang(entity)
    return [
        code for code in EU_OFFICIAL_LANGS
        if code != src and not entity.properties.get(f"title_{code}")
    ]


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
                "source_lang": src_lang,
            },
        )

    def _noop(self, entity: Entity, reason: str, **extra: str) -> Decision:
        return Decision(
            rule_name=self.name, action="noop",
            source_id=entity.id, target_id=entity.id, confidence=0.0,
            entity_type="CohesionProject",
            details={"reason": reason, **extra},
        )
