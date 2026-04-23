"""TranslationEnrichmentContract — contract-side mirror of the Authority rule.

Covers applies() gating, resolve() happy path, fail-soft on transport /
5xx, the per-request `translation_backend` override plumbing, and rule
registration.
"""
from __future__ import annotations

import httpx
import pytest

from src.consolidator.clients.linguistics import (
    EU_OFFICIAL_LANGS,
    LinguisticsClient,
)
from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.contract.enrichment import (
    TranslationEnrichmentContract,
    infer_source_lang,
    missing_targets,
)


pytestmark = pytest.mark.asyncio


def _contract(**props) -> Entity:
    base = {"title": "Neubau eines Baubetriebshofs", "country": "DEU"}
    return Entity(entity_type="Contract", id="TED-1", properties={**base, **props})


# ── pure helpers ──────────────────────────────────────────────────

def test_missing_targets_excludes_source_and_set_langs():
    e = _contract(country="DEU", title_en="Construction of a depot")
    missing = missing_targets(e)
    assert "de" not in missing   # inferred from country=DEU
    assert "en" not in missing   # already present
    assert len(missing) == len(EU_OFFICIAL_LANGS) - 2


def test_infer_source_lang_from_country_then_fallback():
    assert infer_source_lang(_contract(country="FRA")) == "fr"
    assert infer_source_lang(_contract(country="USA")) == "en"
    # Explicit title_lang wins if ETL ever starts writing it.
    assert infer_source_lang(_contract(country="DEU", title_lang="fr")) == "fr"


# ── applies() gating ──────────────────────────────────────────────

async def test_applies_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "src.consolidator.rules.contract.enrichment.settings.linguistics_enabled",
        False,
    )
    rule = TranslationEnrichmentContract()
    assert await rule.applies(_contract()) is False


async def test_applies_skips_when_no_title():
    rule = TranslationEnrichmentContract()
    e = Entity(entity_type="Contract", id="T", properties={"country": "DEU"})
    assert await rule.applies(e) is False


async def test_applies_false_when_complete():
    """All 23 non-source translations already present → nothing to do."""
    rule = TranslationEnrichmentContract()
    props = {"title": "X", "country": "DEU"}
    for lang in EU_OFFICIAL_LANGS:
        if lang != "de":
            props[f"title_{lang}"] = f"[{lang}]X"
    assert await rule.applies(_contract(**props)) is False


async def test_applies_true_when_missing_any_target():
    rule = TranslationEnrichmentContract()
    assert await rule.applies(_contract()) is True


# ── find_candidates self-candidate ─────────────────────────────────

async def test_find_candidates_returns_self_candidate():
    rule = TranslationEnrichmentContract()
    e = _contract()
    cands = await rule.find_candidates(e)
    assert len(cands) == 1
    assert cands[0].entity.id == e.id
    assert cands[0].context == {"enrichment": True}


# ── resolve() with httpx mock ─────────────────────────────────────

def _mock_linguistics(
    translations: dict[str, str] | None = None,
    status: int = 200,
    raise_transport: bool = False,
    capture: dict | None = None,
):
    """Mock /translate — captures the POST body when `capture` dict given."""

    def handler(req: httpx.Request) -> httpx.Response:
        if raise_transport:
            raise httpx.ConnectError("refused")
        if req.url.path.endswith("/translate"):
            if capture is not None:
                import json
                capture["body"] = json.loads(req.content)
            return httpx.Response(status, json={
                "cached": False, "backend": "nllb-local",
                "translations": translations or {},
                "partial_cached_targets": [],
            })
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_resolve_happy_writes_translations_no_embedding(monkeypatch):
    translations = {l: f"[{l}]X" for l in EU_OFFICIAL_LANGS if l != "de"}
    transport = _mock_linguistics(translations=translations)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentContract()
    e = _contract()
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])

    assert decision.action == "enrich"
    assert decision.entity_type == "Contract"
    assert decision.details["field"] == "title"
    assert decision.details["translations"]["en"] == "[en]X"
    assert decision.details["source_lang"] == "de"
    # Contract rule intentionally doesn't compute embeddings (v1 scope).
    assert "embedding" not in decision.details


async def test_resolve_passes_backend_override_from_context(monkeypatch):
    capture: dict = {}
    transport = _mock_linguistics(
        translations={"en": "Construction"}, capture=capture,
    )

    async def _fake_aenter(self):
        capture["backend_at_init"] = self.translation_backend
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentContract()
    e = _contract()
    candidate = Candidate(
        entity=e,
        context={
            "enrichment": True,
            "translation_backend_override": "nllb-local",
        },
    )
    await rule.resolve(e, candidate)

    # The override was applied at client construction, so the outbound
    # /translate payload carries backend="nllb-local".
    assert capture["backend_at_init"] == "nllb-local"
    assert capture["body"]["backend"] == "nllb-local"


async def test_resolve_failsoft_on_transport_error(monkeypatch):
    transport = _mock_linguistics(raise_transport=True)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentContract()
    e = _contract()
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])
    assert decision.action == "noop"
    assert decision.details["reason"] == "linguistics_unavailable"


async def test_resolve_failsoft_on_503(monkeypatch):
    transport = _mock_linguistics(status=503)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentContract()
    e = _contract()
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])
    assert decision.action == "noop"
    assert decision.details["reason"] == "linguistics_unavailable"


# ── registered in the loader ──────────────────────────────────────

async def test_rule_registered_in_loader():
    from src.consolidator.rules.registry import _REGISTRY, list_rules
    from src.consolidator.rules.loader import load_all
    import src.consolidator.rules.loader as L

    _REGISTRY.clear()
    L._loaded = False
    load_all()

    names = [r.name for r in list_rules() if "Contract" in r.entity_types]
    assert "translation_enrichment_contract" in names
