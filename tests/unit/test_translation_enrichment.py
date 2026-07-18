"""TranslationEnrichmentAuthority — applies() gating, resolve() happy path, fail-soft."""
# protected-access: tests reach into the rule's `_client` (the
# linguistics client seam) and the loader's `_loaded` flag —
# both are the private surfaces the enrichment contract pins.
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
    monkeypatch.setattr(
        "src.consolidator.rules.authority.enrichment.settings.linguistics_enabled",
        False,
    )
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
    # no embedding, no translations
    assert await rule.applies(_entity(name="X", name_lang="en")) is True
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
    encoder_id: str | None = "labse@1.0.0-test000",
    status: int = 200,
    raise_transport: bool = False,
):
    """Build an httpx.MockTransport that answers /translate + /embed.

    Pass ``encoder_id=None`` to simulate a misbehaving linguistics service
    that omits the encoder identity — the client treats that as a hard
    error and the rule must propagate it.
    """

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
            body: dict = {
                "cached": False, "backend": "labse-local",
                "dim": len(embedding or [0.1] * 1024),
                "vector": embedding or [0.1] * 1024,
            }
            if encoder_id is not None:
                body["encoder_id"] = encoder_id
            return httpx.Response(status, json=body)
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
    assert decision.details["embedding_encoder"] == "labse@1.0.0-test000"
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


# ── encoder-id handling (signed-mirror supply chain) ──────────────

async def test_embed_returns_vector_and_encoder_id(monkeypatch):
    transport = _mock_linguistics(embedding=[0.1, 0.2, 0.3], encoder_id="labse@1.0.0-abcdef0")

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    async with LinguisticsClient(base_url="http://x") as client:
        vec, enc = await client.embed(text="foo")

    assert vec == [0.1, 0.2, 0.3]
    assert enc == "labse@1.0.0-abcdef0"


async def test_embed_rejects_missing_encoder_id(monkeypatch):
    from src.consolidator.clients.linguistics import LinguisticsError

    transport = _mock_linguistics(embedding=[0.1], encoder_id=None)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    async with LinguisticsClient(base_url="http://x") as client:
        with pytest.raises(LinguisticsError, match="encoder_id"):
            await client.embed(text="foo")


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


# ── configured backend + dim-consistent vectors ───────────────────

async def test_resolve_requests_the_configured_embedding_backend(monkeypatch):
    """The /embed request must carry settings.linguistics_embedding_backend.

    This is the wire-level half of the pipeline-consistency pin in
    tests/unit/test_config.py: the rule asks linguistics for the
    backend the config names, and that backend's dim matches the
    authority_name_embedding_idx vector index.
    """
    import json

    from src.config import EMBEDDING_BACKEND_DIMS, settings
    from src.consolidator.neo4j.migrations import AUTHORITY_NAME_EMBEDDING_DIMS

    seen: dict[str, dict] = {}
    default_backend = settings.linguistics_embedding_backend
    dim = EMBEDDING_BACKEND_DIMS[default_backend]

    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        if req.url.path.endswith("/translate"):
            seen["translate"] = payload
            return httpx.Response(200, json={
                "cached": False, "backend": payload["backend"],
                "translations": {t: f"[{t}]X" for t in payload["targets"]},
                "partial_cached_targets": [],
            })
        if req.url.path.endswith("/embed"):
            seen["embed"] = payload
            return httpx.Response(200, json={
                "cached": False, "backend": payload["backend"],
                "dim": dim, "vector": [0.25] * dim,
                "encoder_id": "labse@1.0.0-836121a",
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentAuthority()
    e = _entity(name="Ministero della Difesa", name_lang="it")
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])

    # Wire contract: the rule requested the configured (default) backend…
    assert seen["embed"]["backend"] == default_backend == "labse-local"
    # …and the vector it writes is dim-consistent with the vector index,
    # so Neo4j will actually index it.
    assert decision.action == "enrich"
    assert len(decision.details["embedding"]) == AUTHORITY_NAME_EMBEDDING_DIMS
    assert decision.details["embedding_encoder"] == "labse@1.0.0-836121a"


async def test_resolve_honours_embedding_backend_override(monkeypatch):
    """CONSOLIDATOR_LINGUISTICS_EMBEDDING_BACKEND=mistral-embed still works —
    the rule forwards whatever backend the settings carry."""
    import json

    monkeypatch.setattr(
        "src.config.settings.linguistics_embedding_backend", "mistral-embed",
    )

    seen: dict[str, dict] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        if req.url.path.endswith("/translate"):
            return httpx.Response(200, json={
                "cached": False, "backend": payload["backend"],
                "translations": {t: f"[{t}]X" for t in payload["targets"]},
                "partial_cached_targets": [],
            })
        if req.url.path.endswith("/embed"):
            seen["embed"] = payload
            return httpx.Response(200, json={
                "cached": False, "backend": payload["backend"],
                "dim": 1024, "vector": [0.25] * 1024,
                "encoder_id": "mistral-embed@api-mistral-embed-2312",
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)

    rule = TranslationEnrichmentAuthority()
    e = _entity(name="X", name_lang="en")
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])

    assert seen["embed"]["backend"] == "mistral-embed"
    assert decision.details["embedding_encoder"] == (
        "mistral-embed@api-mistral-embed-2312"
    )
