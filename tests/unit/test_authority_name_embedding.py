"""AuthorityNameEmbedding — the vector the cross-lingual similarity rule
compares. It embeds and nothing else: translating names is fontem-translator's."""
# protected-access: the loader's `_loaded` flag is reset to reload rules.
# import-outside-toplevel: registry / loader are imported inside test bodies
# so per-test patches activate before module-import side effects.
# pylint: disable=protected-access,import-outside-toplevel
from __future__ import annotations

import json

import httpx
import pytest

from src.config import EMBEDDING_BACKEND_DIMS, settings
from src.consolidator.clients.linguistics import LinguisticsClient, LinguisticsError
from src.consolidator.neo4j.migrations import AUTHORITY_NAME_EMBEDDING_DIMS
from src.consolidator.rules.authority.name_embedding import (
    AuthorityNameEmbedding,
    needs_embedding,
)
from src.consolidator.rules.base import Entity

pytestmark = pytest.mark.asyncio

DIM = EMBEDDING_BACKEND_DIMS[settings.linguistics_embedding_backend]


def _entity(**props) -> Entity:
    return Entity(entity_type="Authority", id="AUTH-1", properties={"name": "X", **props})


def _serve(monkeypatch, *, status=200, encoder_id="labse@1.0.0-test000", seen=None,
           raise_transport=False):
    def handler(req: httpx.Request) -> httpx.Response:
        if raise_transport:
            raise httpx.ConnectError("refused")
        if seen is not None:
            seen.append((req.url.path, json.loads(req.content)))
        if not req.url.path.endswith("/embed"):
            return httpx.Response(404)
        body = {"cached": False, "backend": "labse-local", "dim": DIM, "vector": [0.25] * DIM}
        if encoder_id is not None:
            body["encoder_id"] = encoder_id
        return httpx.Response(status, json=body)

    transport = httpx.MockTransport(handler)

    async def _fake_aenter(self):
        self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url)
        return self

    monkeypatch.setattr(LinguisticsClient, "__aenter__", _fake_aenter)


async def test_a_name_without_a_vector_of_the_right_dimension_needs_one():
    assert needs_embedding(_entity())
    assert needs_embedding(_entity(name_embedding=[]))
    assert needs_embedding(_entity(name_embedding=[0.1] * 1024))      # another encoder
    assert not needs_embedding(_entity(name_embedding=[0.1] * DIM))


async def test_it_applies_only_when_there_is_a_name_to_embed(monkeypatch):
    rule = AuthorityNameEmbedding()
    assert await rule.applies(_entity())
    assert not await rule.applies(_entity(name=""))
    assert not await rule.applies(_entity(name_embedding=[0.1] * DIM))
    monkeypatch.setattr(settings, "linguistics_enabled", False)
    assert not await rule.applies(_entity())


async def test_it_embeds_the_name_and_never_asks_for_a_translation(monkeypatch):
    seen: list = []
    _serve(monkeypatch, seen=seen)
    rule = AuthorityNameEmbedding()
    e = _entity(name="Ministero della Difesa")
    (candidate,) = await rule.find_candidates(e)
    assert candidate.entity.id == e.id
    decision = await rule.resolve(e, candidate)
    assert decision.action == "enrich" and decision.source_id == decision.target_id == e.id
    assert len(decision.details["embedding"]) == AUTHORITY_NAME_EMBEDDING_DIMS
    assert decision.details["embedding_encoder"] == "labse@1.0.0-test000"
    assert "translations" not in decision.details
    assert [path for path, _ in seen] == ["/embed"]
    assert seen[0][1]["backend"] == settings.linguistics_embedding_backend


@pytest.mark.parametrize("serve", [{"raise_transport": True}, {"status": 503}])
async def test_linguistics_being_down_is_a_noop_retried_later(monkeypatch, serve):
    _serve(monkeypatch, **serve)
    rule = AuthorityNameEmbedding()
    e = _entity()
    decision = await rule.resolve(e, (await rule.find_candidates(e))[0])
    assert decision.action == "noop" and decision.details["reason"] == "LinguisticsUnavailable"


async def test_a_vector_without_its_encoder_is_refused(monkeypatch):
    _serve(monkeypatch, encoder_id=None)
    async with LinguisticsClient(base_url="http://x") as client:
        with pytest.raises(LinguisticsError, match="encoder_id"):
            await client.embed(text="foo")


async def test_it_runs_before_the_rule_that_compares_the_vectors():
    from src.consolidator.rules.registry import _REGISTRY, list_rules
    from src.consolidator.rules.loader import load_all
    import src.consolidator.rules.loader as L

    _REGISTRY.clear()
    L._loaded = False
    load_all()
    names = [r.name for r in list_rules() if "Authority" in r.entity_types]
    assert names.index("authority_name_embedding") < names.index("embedding_cosine_authority")
    assert not any("translation" in r.name for r in list_rules())
