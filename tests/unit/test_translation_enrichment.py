"""TranslationEnrichmentAuthority — applies() gating, resolve() happy path, fail-soft."""
from __future__ import annotations

import httpx
import pytest

from src.consolidator.clients import linguistics as ling_mod
from src.consolidator.clients.linguistics import (
    EU_OFFICIAL_LANGS,
    LinguisticsClient,
    LinguisticsUnavailable,
)
from src.consolidator.rules.authority.enrichment import (
    TranslationEnrichmentAuthority,
    infer_source_lang,
    missing_targets,
    needs_embedding,
)
from src.consolidator.rules.base import Entity


pytestmark = pytest.mark.asyncio


def _entity(**props) -> Entity:
    return Entity(entity_type="Authority", id="AUTH-1", properties={"name": "X", **props})


# ── pure helpers ──────────────────────────────────────────────────

def test_missing_targets_excludes_source_lang_and_set_langs():
    e = _entity(name="Ministero della Difesa", name_lang="it", name_en="Ministry")
    missing = missing_targets(e)
    assert "it" not in missing               # source lang never a target
    assert "en" not in missing               # already set
    assert len(missing) == len(EU_OFFICIAL_LANGS) - 2


def test_missing_targets_when_nothing_set():
    e = _entity(name="X", name_lang="fr")
    missing = missing_targets(e)
    # All 24 minus the source = 23
    assert len(missing) == len(EU_OFFICIAL_LANGS) - 1


def test_needs_embedding_detects_absence_and_empty():
    assert needs_embedding(_entity()) is True
    assert needs_embedding(_entity(name_embedding=[])) is True
    assert needs_embedding(_entity(name_embedding=[0.1, 0.2])) is False


def test_infer_source_lang_prefers_explicit():
    assert infer_source_lang(_entity(name_lang="pl", country="POL")) == "pl"
    # Mixed-case is normalised
    assert infer_source_lang(_entity(name_lang="PL")) == "pl"


def test_infer_source_lang_falls_back_to_country_primary():
    # No explicit name_lang → country's primary EU official language
    assert infer_source_lang(_entity(country="POL")) == "pl"
    assert infer_source_lang(_entity(country="DEU")) == "de"
    assert infer_source_lang(_entity(country="BEL")) == "nl"
    assert infer_source_lang(_entity(country="MLT")) == "mt"


def test_infer_source_lang_defaults_to_en_for_unknown_country():
    assert infer_source_lang(_entity(country="USA")) == "en"
    assert infer_source_lang(_entity()) == "en"


# ── applies() gating ──────────────────────────────────────────────

async def test_applies_skips_when_disabled(monkeypatch):
    monkeypatch.setattr("src.consolidator.rules.authority.enrichment.settings.linguistics_enabled", False)
    rule = TranslationEnrichmentAuthority()
    assert await rule.applies(_entity(name="X")) is False


async def test_applies_skips_when_no_name():
    rule = TranslationEnrichmentAuthority()
    # The default _entity helper sets name="X"; override with falsy.
    e = Entity(entity_type="Authority", id="A", properties={"name": ""})
    assert await rule.applies(e) is False


async def test_applies_skips_when_complete():
    rule = TranslationEnrichmentAuthority()
    props = {"name": "X", "name_lang": "it", "name_embedding": [0.1] * 4}
    for lang in EU_OFFICIAL_LANGS:
        if lang != "it":
            props[f"name_{lang}"] = f"[{lang}]X"
    assert await rule.applies(_entity(**props)) is False


async def test_applies_true_when_missing_translation_or_embedding():
    rule = TranslationEnrichmentAuthority()
    assert await rule.applies(_entity(name="X", name_lang="en")) is True     # no embedding, no translations
    assert await rule.applies(
        _entity(name="X", name_lang="en", name_embedding=[0.1])   # has embed, no translations
    ) is True


# ── find_candidates returns self-candidate ────────────────────────

async def test_find_candidates_returns_self_candidate():
    rule = TranslationEnrichmentAuthority()
    e = _entity(name="X", name_lang="en")
    cands = await rule.find_candidates(e)
    assert len(cands) == 1
    assert cands[0].entity.id == e.id
    assert cands[0].context == {"enrichment": True}


# ── resolve() with httpx mock ─────────────────────────────────────

def _mock_linguistics(
    translations: dict[str, str] | None = None,
    embedding: list[float] | None = None,
    status: int = 200,
    raise_transport: bool = False,
):
    """Build an httpx.MockTransport that answers /translate + /embed."""

    def handler(req: httpx.Request) -> httpx.Response:
        if raise_transport:
            raise httpx.ConnectError("refused")
        if req.url.path.endswith("/translate"):
            return httpx.Response(status, json={
                "cached": False, "backend": "mistral",
                "translations": translations or {},
                "partial_cached_targets": [],
            })
        if req.url.path.endswith("/embed"):
            return httpx.Response(status, json={
                "cached": False, "backend": "mistral-embed",
                "dim": len(embedding or [0.1] * 1024),
                "vector": embedding or [0.1] * 1024,
            })
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_resolve_happy_writes_translations_and_embedding(monkeypatch):
    translations = {lang: f"[{lang}]X" for lang in EU_OFFICIAL_LANGS if lang != "it"}
    transport = _mock_linguistics(translations=translations, embedding=[0.5] * 8)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentAuthority()
    e = _entity(name="Ministero della Difesa", name_lang="it")
    candidates = await rule.find_candidates(e)
    decision = await rule.resolve(e, candidates[0])

    assert decision.action == "enrich"
    assert decision.source_id == decision.target_id == e.id
    assert decision.details["translations"]["en"] == "[en]X"
    assert decision.details["embedding"] == [0.5] * 8
    assert decision.details["source_lang"] == "it"


async def test_resolve_failsoft_on_transport_error(monkeypatch):
    transport = _mock_linguistics(raise_transport=True)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentAuthority()
    e = _entity(name="X", name_lang="en")
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])
    assert decision.action == "noop"
    assert decision.details["reason"] == "linguistics_unavailable"


async def test_resolve_failsoft_on_503(monkeypatch):
    transport = _mock_linguistics(status=503)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentAuthority()
    e = _entity(name="X", name_lang="en")
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])
    assert decision.action == "noop"
    assert decision.details["reason"] == "linguistics_unavailable"


# ── registered in loader at the top of Authority queue ────────────

async def test_rule_registered_first_among_authority():
    from src.consolidator.rules.registry import _REGISTRY
    from src.consolidator.rules.loader import load_all
    import src.consolidator.rules.loader as L

    _REGISTRY.clear()
    L._loaded = False
    load_all()

    from src.consolidator.rules.registry import list_rules
    names = [r.name for r in list_rules() if "Authority" in r.entity_types]
    # Enrichment is present and precedes the match/merge rules for Authority.
    assert "translation_enrichment_authority" in names
    assert names.index("translation_enrichment_authority") < names.index(
        "exact_authority_id_match"
    )
