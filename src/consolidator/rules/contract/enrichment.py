"""TranslationEnrichmentContract — fills in missing title_<lang> on :Contract.

Mirrors TranslationEnrichmentAuthority but writes translated titles on
contracts instead of names on authorities. v1 scope: title only — no
`description` (length often exceeds NLLB's 256-token window) and no
embedding (171k+ contracts × 1024 floats is too costly for the value).
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
    """Best-effort source language for a contract's title.

    TED doesn't carry a title_lang property today, so we infer from the
    buyer's country code. Falls through to "en" for unknowns.
    """
    explicit = (entity.properties.get("title_lang") or "").lower()
    if explicit:
        return explicit
    return source_lang_from_country(entity.properties.get("country"))


def missing_targets(entity: Entity) -> list[str]:
    """Return EU locales that have no title_<lang> on the node yet."""
    src = infer_source_lang(entity)
    return [
        code for code in EU_OFFICIAL_LANGS
        if code != src and not entity.properties.get(f"title_{code}")
    ]


class TranslationEnrichmentContract(Rule):
    name = "translation_enrichment_contract"
    description = (
        "Fill missing EU-language translations (title_<lang>) on :Contract "
        "by calling gmr-linguistics. Runs per-entity; never merges or flags. "
        "v1: title only — description translation is deferred."
    )
    entity_types = {"Contract"}
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
                "translation_enrichment_contract: linguistics unavailable for {id}: {exc}",
                id=entity.id, exc=exc,
            )
            return Decision(
                rule_name=self.name, action="noop",
                source_id=entity.id, target_id=entity.id, confidence=0.0,
                entity_type="Contract",
                details={"reason": "linguistics_unavailable"},
            )
        except LinguisticsError as exc:
            logger.error(
                "translation_enrichment_contract: linguistics hard error for {id}: {exc}",
                id=entity.id, exc=exc,
            )
            return Decision(
                rule_name=self.name, action="noop",
                source_id=entity.id, target_id=entity.id, confidence=0.0,
                entity_type="Contract",
                details={"reason": "linguistics_error", "message": str(exc)},
            )

        if not translations:
            return Decision(
                rule_name=self.name, action="noop",
                source_id=entity.id, target_id=entity.id, confidence=0.0,
                entity_type="Contract", details={"reason": "already_complete"},
            )

        return Decision(
            rule_name=self.name,
            action="enrich",
            source_id=entity.id,
            target_id=entity.id,
            confidence=1.0,
            entity_type="Contract",
            details={
                "field": "title",
                "translations": translations,
                "source_lang": src_lang,
            },
        )
