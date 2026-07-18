"""Pipeline-consistency pins for the embedding defaults.

The exact regression that killed embedding similarity in prod: the
default linguistics_embedding_backend was mistral-embed (1024-d) while
the authority_name_embedding_idx vector index was declared 768-d.
Neo4j does not error on the mismatch — it silently skips indexing the
vectors — so enrichment "worked", the index stayed empty, and zero
authorities ever matched. These tests pin default backend == known dim
== index dim so that any future drift fails CI instead of prod.
"""
from __future__ import annotations

from src.config import EMBEDDING_BACKEND_DIMS, Settings, settings
from src.consolidator.clients.linguistics import LinguisticsClient
from src.consolidator.neo4j import migrations


# ── default backend ↔ vector index dims (the DOA regression) ─────────

def test_default_embedding_backend_is_mistral_embed():
    """labse-local was the intended default (#189) but the signed model
    mirror ships only config/pooling for labse-1.0.0 — no weights — so
    the backend 500s on every call (verified in prod 2026-07-18).
    mistral-embed works today, costs cents for short names, and the
    encoder_id stamp keeps a future labse migration clean."""
    assert Settings().linguistics_embedding_backend == "mistral-embed"


def test_default_backend_dim_matches_authority_vector_index():
    default_backend = Settings().linguistics_embedding_backend
    assert default_backend in EMBEDDING_BACKEND_DIMS, (
        f"default backend {default_backend!r} has no known dim — add it to "
        "EMBEDDING_BACKEND_DIMS (mirror of the fontem-linguistics catalog)"
    )
    assert (
        EMBEDDING_BACKEND_DIMS[default_backend]
        == migrations.AUTHORITY_NAME_EMBEDDING_DIMS
    ), (
        "default embedding backend produces vectors of a different dim than "
        "authority_name_embedding_idx — Neo4j will silently not index them "
        "and embedding_cosine_authority will never find a candidate"
    )


def test_vector_index_cypher_declares_the_pinned_dims():
    stmt = next(
        s for s in migrations.INDEX_CYPHER
        if "authority_name_embedding_idx" in s
    )
    expected = f"`vector.dimensions`: {migrations.AUTHORITY_NAME_EMBEDDING_DIMS}"
    assert expected in stmt


def test_linguistics_client_default_backend_matches_settings_default():
    # The client dataclass default and the Settings default must agree,
    # or a caller constructing the client without the setting silently
    # requests a different vector space.
    assert LinguisticsClient(base_url="http://x").embedding_backend == (
        Settings().linguistics_embedding_backend
    )


def test_mistral_stays_available_via_env_override(monkeypatch):
    monkeypatch.setenv(
        "CONSOLIDATOR_LINGUISTICS_EMBEDDING_BACKEND", "mistral-embed",
    )
    assert Settings().linguistics_embedding_backend == "mistral-embed"
    # Its dim is known too, so an operator can provision a matching index.
    assert EMBEDDING_BACKEND_DIMS["mistral-embed"] == 1024


# ── stale-namespace defaults ─────────────────────────────────────────

def test_neo4j_default_uri_points_at_fontem_prod():
    uri = Settings().neo4j_uri
    assert uri == "bolt://neo4j.fontem-prod.svc.cluster.local:7687"
    assert ".gmr.svc" not in uri  # the pre-rename namespace


def test_no_default_references_the_retired_gmr_namespace():
    defaults = Settings().model_dump()
    stale = {
        k: v for k, v in defaults.items()
        if isinstance(v, str) and ".gmr.svc" in v
    }
    assert not stale, f"stale gmr-namespace defaults: {stale}"


def test_module_singleton_uses_the_same_defaults():
    # `settings` is instantiated at import time; guard against a stale
    # singleton diverging from the class defaults under test.
    assert settings.linguistics_embedding_backend in EMBEDDING_BACKEND_DIMS


def test_translation_default_is_mistral():
    """Measured on prod 2026-07-18: one Mistral chat call translates all
    23 target languages per authority (~166k one-time calls under the
    linguistics service's $50/day spend cap), while a 23-target
    nllb-local request takes 182s on CPU and 502s — months of wall
    clock for the same backfill. The paid-but-capped API is the design;
    nllb-local stays reachable via
    CONSOLIDATOR_LINGUISTICS_TRANSLATION_BACKEND for offline use."""
    assert Settings().linguistics_translation_backend == "mistral"


def test_nllb_translation_stays_available_via_env_override(monkeypatch):
    monkeypatch.setenv(
        "CONSOLIDATOR_LINGUISTICS_TRANSLATION_BACKEND", "nllb-local",
    )
    assert Settings().linguistics_translation_backend == "nllb-local"
