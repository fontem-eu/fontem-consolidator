"""Tests for EmbeddingCosineSameAuthority — applies() gating, vector-
index query shape, Decision mapping, registered-in-loader, encoder
allowlist.
"""
# protected-access: the loader's `_loaded` / registry `_REGISTRY`
# are reset between tests so the per-test rule set is deterministic.
# import-outside-toplevel: registry / loader are imported inside
# tests so the reset happens before module-import side effects.
# too-few-public-methods: in-test stub classes deliberately expose
# only the attributes the rule reads.
# pylint: disable=protected-access,import-outside-toplevel,too-few-public-methods
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import settings as _settings
from src.consolidator.rules.authority.embedding_similarity import (
    EmbeddingCosineSameAuthority,
)
from src.consolidator.rules.base import Candidate, Entity


pytestmark = pytest.mark.asyncio

VEC = [0.1, 0.2, 0.3] + [0.0] * 765  # 768-d placeholder
ENC = "labse@1.0.0-836121a"


def _authority(**props) -> Entity:
    base = {
        "name": "Ministry of Defence",
        "country": "IRL",
        "name_embedding": VEC,
        "name_embedding_encoder": ENC,
        "name_embedding_dim": 768,
    }
    base.update(props)
    return Entity(entity_type="Authority", id="AUTH-1", properties=base)


# ── applies() ─────────────────────────────────────────────────────────

async def test_applies_true_when_embedding_and_accepted_encoder(monkeypatch):
    monkeypatch.setattr(_settings, "embedding_cosine_enabled", True)
    monkeypatch.setattr(
        _settings, "embedding_cosine_accepted_encoders", ENC,
    )
    rule = EmbeddingCosineSameAuthority()
    assert await rule.applies(_authority()) is True


async def test_applies_false_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(_settings, "embedding_cosine_enabled", False)
    rule = EmbeddingCosineSameAuthority()
    assert await rule.applies(_authority()) is False


async def test_applies_false_when_no_embedding(monkeypatch):
    monkeypatch.setattr(_settings, "embedding_cosine_enabled", True)
    rule = EmbeddingCosineSameAuthority()
    e = _authority()
    e.properties.pop("name_embedding")
    assert await rule.applies(e) is False


async def test_applies_false_when_encoder_not_accepted(monkeypatch):
    monkeypatch.setattr(_settings, "embedding_cosine_enabled", True)
    monkeypatch.setattr(
        _settings, "embedding_cosine_accepted_encoders", "labse@2.0.0-xxxxxxx",
    )
    rule = EmbeddingCosineSameAuthority()
    # Current encoder is labse@1.0.0-... — not on the allowlist
    assert await rule.applies(_authority()) is False


async def test_applies_false_for_un_versioned_legacy_embedding(monkeypatch):
    """Rows from before the versioning migration must abstain."""
    monkeypatch.setattr(_settings, "embedding_cosine_enabled", True)
    monkeypatch.setattr(
        _settings, "embedding_cosine_accepted_encoders", ENC,
    )
    rule = EmbeddingCosineSameAuthority()
    e = _authority(name_embedding_encoder="mistral-embed@legacy-pre-versioning")
    assert await rule.applies(e) is False


# ── find_candidates() — vector-index shape + result mapping ───────────

def _mock_driver_with_records(records: list[dict]) -> MagicMock:
    """Patch-in driver returning `records` for the vector-index query."""
    # Each record is accessed as rec["n"] / rec["s"]
    fake_records = [MagicMock(__getitem__=lambda self, k, r=r: r[k]) for r in records]

    async def _aiter():
        for r in fake_records:
            yield r

    # The rule does `[rec async for rec in result]` — so result must be
    # an async iterable. AsyncMock alone doesn't implement __aiter__.
    class _Result:
        def __aiter__(self):
            return _aiter()

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.run = AsyncMock(return_value=_Result())

    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver, session


async def test_find_candidates_maps_vector_index_results(monkeypatch):
    monkeypatch.setattr(_settings, "embedding_cosine_top_k", 3)
    monkeypatch.setattr(_settings, "embedding_cosine_threshold", 0.90)
    monkeypatch.setattr(_settings, "embedding_cosine_jaro_winkler_min", 0.30)
    monkeypatch.setattr(_settings, "embedding_cosine_cross_country_only", True)

    # Two candidate rows returned from the vector index (cosine-filtered +
    # JW-filtered + cross-country-filtered at the database layer).
    node_a = {
        "authority_id": "AUTH-42", "name": "Ministero della Difesa",
        "country": "ITA", "name_embedding_encoder": ENC,
    }
    node_b = {
        "authority_id": "AUTH-99", "name": "Ministère de la Défense",
        "country": "FRA", "name_embedding_encoder": ENC,
    }
    records = [
        {"n": node_a, "s": 0.94},
        {"n": node_b, "s": 0.91},
    ]
    driver, session = _mock_driver_with_records(records)

    async def _fake_get_driver():
        return driver
    monkeypatch.setattr(
        "src.consolidator.neo4j.client.get_driver", _fake_get_driver,
    )

    rule = EmbeddingCosineSameAuthority()
    cands = await rule.find_candidates(_authority())

    assert len(cands) == 2
    assert [c.entity.id for c in cands] == ["AUTH-42", "AUTH-99"]
    assert cands[0].context["cosine_score"] == 0.94
    assert cands[1].context["cosine_score"] == 0.91

    # Vector-index query was built with the right knobs.
    # Jaro-Winkler post-filter is applied in Python (rapidfuzz), not in
    # Cypher — APOC ships no jaroWinkler function.
    args, kwargs = session.run.call_args
    assert "db.index.vector.queryNodes" in args[0]
    assert "apoc.text.jaroWinkler" not in args[0]
    assert "cross_country_only" in args[0]
    assert kwargs["k"] == 4  # top_k (3) + 1 for self-exclusion headroom
    assert kwargs["vec"] == VEC
    assert kwargs["threshold"] == 0.90
    assert kwargs["enc"] == ENC
    assert kwargs["cross_country_only"] is True
    assert kwargs["self_country"] == "IRL"
    # self_name + jw_min are not query params anymore — JW happens in Python
    assert "jw_min" not in kwargs
    assert "self_name" not in kwargs


async def test_find_candidates_jw_post_filter_drops_low_overlap(monkeypatch):
    """JW post-filter rejects pairs whose raw names share little
    surface form. Useful as an extra knob (default permissive); set
    via env when LaBSE role-matches start dominating."""
    monkeypatch.setattr(_settings, "embedding_cosine_top_k", 3)
    monkeypatch.setattr(_settings, "embedding_cosine_threshold", 0.90)
    # In-test threshold is deliberately tighter than the package
    # default (0.30) so the assertion is meaningful on the test pairs.
    monkeypatch.setattr(_settings, "embedding_cosine_jaro_winkler_min", 0.70)
    monkeypatch.setattr(_settings, "embedding_cosine_cross_country_only", True)

    # JW("ministry of defence", "ministero della difesa")  ≈ 0.85  → keep
    # JW("ministry of defence", "rete ferroviaria s.p.a.")  ≈ 0.42 → drop
    # JW("ministry of defence", "department of defence")    ≈ 0.68 → drop
    keep_lookalike = {
        "authority_id": "AUTH-42", "name": "Ministero della Difesa",
        "country": "ITA", "name_embedding_encoder": ENC,
    }
    drop_role_only = {
        "authority_id": "AUTH-99", "name": "Rete Ferroviaria S.p.A.",
        "country": "ITA", "name_embedding_encoder": ENC,
    }
    drop_partial = {
        "authority_id": "AUTH-7", "name": "Department of Defence",
        "country": "GBR", "name_embedding_encoder": ENC,
    }
    records = [
        {"n": keep_lookalike, "s": 0.93},
        {"n": drop_role_only, "s": 0.92},
        {"n": drop_partial, "s": 0.91},
    ]
    driver, _ = _mock_driver_with_records(records)

    async def _fake_get_driver():
        return driver
    monkeypatch.setattr(
        "src.consolidator.neo4j.client.get_driver", _fake_get_driver,
    )

    rule = EmbeddingCosineSameAuthority()
    cands = await rule.find_candidates(_authority())
    kept_ids = {c.entity.id for c in cands}
    assert kept_ids == {"AUTH-42"}, (
        f"only the cross-lingual lookalike should survive at JW>=0.70, got {kept_ids}"
    )


async def test_find_candidates_empty_when_index_returns_nothing(monkeypatch):
    driver, _ = _mock_driver_with_records([])

    async def _fake_get_driver():
        return driver
    monkeypatch.setattr(
        "src.consolidator.neo4j.client.get_driver", _fake_get_driver,
    )

    rule = EmbeddingCosineSameAuthority()
    assert await rule.find_candidates(_authority()) == []


# ── resolve() ────────────────────────────────────────────────────────

async def test_resolve_emits_flag_with_cosine_as_confidence():
    rule = EmbeddingCosineSameAuthority()
    e = _authority()
    cand = Candidate(
        entity=_authority(name="Ministero della Difesa", country="ITA"),
        context={"cosine_score": 0.89},
    )
    d = await rule.resolve(e, cand)

    assert d.action == "flag"
    assert d.rule_name == "embedding_cosine_authority"
    assert d.confidence == 0.89
    assert d.source_id == e.id
    assert d.target_id == cand.entity.id
    assert d.details["method"] == "embedding_cosine"
    assert d.details["cosine_score"] == 0.89
    assert d.details["encoder_id"] == ENC
    assert d.details["source_country"] == "IRL"
    assert d.details["target_country"] == "ITA"


# ── registered in loader ─────────────────────────────────────────────

async def test_rule_registered_in_loader():
    from src.consolidator.rules.registry import _REGISTRY, list_rules
    from src.consolidator.rules.loader import load_all
    import src.consolidator.rules.loader as _L

    _REGISTRY.clear()
    _L._loaded = False
    load_all()

    by_name = {r.name: r for r in list_rules() if "Authority" in r.entity_types}
    assert "embedding_cosine_authority" in by_name

    # Engine runs rules in confidence-descending order. Embedding cosine
    # (0.87) runs AFTER the 0.9+ deterministic rules (exact-id, exact-
    # name any/same country, same-country fuzzy), i.e. only after each
    # of those has had first crack at the pair. Before GDS node
    # similarity (0.80), which is a broader-net graph-structure signal.
    names = [r.name for r in list_rules() if "Authority" in r.entity_types]
    assert names.index("embedding_cosine_authority") > names.index(
        "exact_name_any_country_authority"
    )
    assert names.index("embedding_cosine_authority") < names.index(
        "gds_node_similarity_authority"
    )
