"""TranslationEnrichmentContract — contract-side mirror of the Authority rule.

Covers applies() gating, resolve() happy path, fail-soft on transport /
5xx, the per-request `translation_backend` override plumbing, and rule
registration.
"""
# protected-access: tests reach into the rule's `_client` and the
# loader's `_loaded` / registry `_REGISTRY` — the private surfaces
# the enrichment contract pins.
# import-outside-toplevel: registry / loader / clients are
# imported inside test bodies so per-test patches activate
# before module-import side effects.
# pylint: disable=protected-access,import-outside-toplevel
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
    """A canonical contract above the live cut-off whose notice states German.
    Pass a key as None to drop it."""
    base = {"title": "Neubau eines Baubetriebshofs", "country": "DEU",
            "title_lang": "de", "value_eur": 300_000_000.0, "ted_notice_id": "123456-2024"}
    merged = {k: v for k, v in {**base, **props}.items() if v is not None}
    return Entity(entity_type="Contract", id="123456-2024", properties=merged)


# ── pure helpers ──────────────────────────────────────────────────

def test_missing_targets_excludes_source_and_set_langs():
    e = _contract(title_en="Construction of a depot")
    missing = missing_targets(e)
    assert "de" not in missing   # the notice states German
    assert "en" not in missing   # already present
    assert len(missing) == len(EU_OFFICIAL_LANGS) - 2


def test_the_source_is_the_notice_then_a_kept_detection_never_the_country():
    assert infer_source_lang(_contract(title_lang="fr")) == "fr"
    assert infer_source_lang(_contract(title_lang=None, title_lang_detected="it")) == "it"
    assert infer_source_lang(_contract(title_lang=None, country="FRA")) is None
    assert infer_source_lang(_contract(title_lang="no")) == "und"    # outside the 24
    assert infer_source_lang(_contract(title_lang=None, title_lang_detected="und")) is None


def test_a_changed_title_is_translated_again_in_full():
    e = _contract(title_en="Old", title_fr="Ancien", title_translated_from="Alter Titel")
    assert len(missing_targets(e)) == len(EU_OFFICIAL_LANGS) - 1
    same = _contract(title_en="Old", title_translated_from="Neubau eines Baubetriebshofs")
    assert "en" not in missing_targets(same)


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


async def test_live_translation_stops_below_the_cut_off():
    """Below EUR 250M the budgeted backfill decides, not the live rule."""
    rule = TranslationEnrichmentContract()
    assert await rule.applies(_contract(value_eur=249_999_999.0)) is False
    assert await rule.applies(_contract(value_eur=None)) is False
    assert await rule.applies(_contract(value_eur=250_000_000.0)) is True


async def test_a_legacy_oj_s_twin_is_not_translated_live():
    rule = TranslationEnrichmentContract()
    assert await rule.applies(_contract(ted_notice_id="2021/S 129-344226")) is False


async def test_a_title_in_no_known_language_waits_for_the_backfill():
    """The backfill runner detects before it translates; the live rule
    never guesses from the buyer's country."""
    rule = TranslationEnrichmentContract()
    assert await rule.applies(_contract(title_lang=None)) is False


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
    # The rule translated from "de" but claims no title_lang: the loader owns it.
    assert decision.details["source_lang"] is None
    # What it was translated from, so a later change of title is seen.
    assert decision.details["translated_from"] == "Neubau eines Baubetriebshofs"
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
