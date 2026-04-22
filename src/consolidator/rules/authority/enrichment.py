"""TranslationEnrichmentRule — fills in missing name_<lang> + name_embedding on :Authority.

Per-entity enrichment (not pairwise). The engine treats `action == "enrich"`
as a signal that a self-targeted Decision is legitimate, and dispatches to
the `_enrich` executor which writes the computed properties back to the node.

Fails soft: if gmr-linguistics is unreachable or rate-limited, the rule emits
no decisions and the consolidation run records a noop for it. A later run
retries.
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


# Country → primary official language. Used when the Authority node doesn't
# carry a `name_lang` yet (ETL hasn't backfilled it). Covers the 27 EU member
# states in alpha-3; non-EU buyers (UK/CH/US embassies appearing in TED)
# fall through to "en".
_COUNTRY_PRIMARY_LANG: dict[str, str] = {
    "AUT": "de", "BEL": "nl", "BGR": "bg", "HRV": "hr", "CYP": "el",
    "CZE": "cs", "DNK": "da", "EST": "et", "FIN": "fi", "FRA": "fr",
    "DEU": "de", "GRC": "el", "HUN": "hu", "IRL": "en", "ITA": "it",
    "LVA": "lv", "LTU": "lt", "LUX": "fr", "MLT": "mt", "NLD": "nl",
    "POL": "pl", "PRT": "pt", "ROU": "ro", "SVK": "sk", "SVN": "sl",
    "ESP": "es", "SWE": "sv",
}


def infer_source_lang(entity: Entity) -> str:
    """Source language for translation prompts.

    Order: explicit `name_lang` → country's primary official language → "en".
    Without this fallback Polish/Czech/etc. authorities get treated as English
    and Mistral returns the original string untranslated (garbage in, garbage
    out).
    """
    explicit = (entity.properties.get("name_lang") or "").lower()
    if explicit:
        return explicit
    country = (entity.properties.get("country") or "").upper()
    return _COUNTRY_PRIMARY_LANG.get(country, "en")


def missing_targets(entity: Entity) -> list[str]:
    """Return the EU locales that have no name_<lang> set on the node."""
    # Never translate a name into its own language — Mistral would round-trip
    # the original and burn tokens for no benefit.
    src = infer_source_lang(entity)
    return [
        code for code in EU_OFFICIAL_LANGS
        if code != src and not entity.properties.get(f"name_{code}")
    ]


def needs_embedding(entity: Entity) -> bool:
    vec = entity.properties.get("name_embedding")
    return not isinstance(vec, list) or len(vec) == 0


class TranslationEnrichmentAuthority(Rule):
    name = "translation_enrichment_authority"
    description = (
        "Fill missing EU-language translations (name_<lang>) and the "
        "name_embedding vector on :Authority by calling gmr-linguistics. "
        "Runs per-entity; never merges or flags."
    )
    entity_types = {"Authority"}
    confidence = 1.0
    action = "enrich"

    async def applies(self, entity: Entity) -> bool:
        if not settings.linguistics_enabled:
            return False
        if not entity.properties.get("name"):
            return False
        return bool(missing_targets(entity)) or needs_embedding(entity)

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        # Per-entity rule: the candidate is the entity itself. The engine
        # special-cases self-candidates for action == "enrich" rules.
        return [Candidate(entity=entity, context={"enrichment": True})]

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        name = entity.properties["name"]
        src_lang = infer_source_lang(entity)

        targets = missing_targets(entity)
        translations: dict[str, str] = {}
        embedding: list[float] | None = None

        try:
            async with LinguisticsClient(
                base_url=settings.linguistics_url,
                timeout_s=settings.linguistics_timeout_s,
                translation_backend=settings.linguistics_translation_backend,
                embedding_backend=settings.linguistics_embedding_backend,
            ) as client:
                if targets:
                    translations = await client.translate(
                        text=name, source_lang=src_lang, targets=targets,
                    )
                if needs_embedding(entity):
                    embedding = await client.embed(text=name)
        except LinguisticsUnavailable as exc:
            logger.warning(
                "translation_enrichment: linguistics unavailable for {id}: {exc}",
                id=entity.id, exc=exc,
            )
            return Decision(
                rule_name=self.name, action="noop",
                source_id=entity.id, target_id=entity.id, confidence=0.0,
                entity_type="Authority",
                details={"reason": "linguistics_unavailable"},
            )
        except LinguisticsError as exc:
            logger.error(
                "translation_enrichment: linguistics hard error for {id}: {exc}",
                id=entity.id, exc=exc,
            )
            return Decision(
                rule_name=self.name, action="noop",
                source_id=entity.id, target_id=entity.id, confidence=0.0,
                entity_type="Authority",
                details={"reason": "linguistics_error", "message": str(exc)},
            )

        if not translations and embedding is None:
            # Nothing to do — entity was already complete between applies() and here.
            return Decision(
                rule_name=self.name, action="noop",
                source_id=entity.id, target_id=entity.id, confidence=0.0,
                entity_type="Authority", details={"reason": "already_complete"},
            )

        return Decision(
            rule_name=self.name,
            action="enrich",
            source_id=entity.id,
            target_id=entity.id,  # self-enrichment
            confidence=1.0,
            entity_type="Authority",
            details={
                "translations": translations,
                "embedding": embedding,
                "source_lang": src_lang,
            },
        )
