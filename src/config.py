from pydantic_settings import BaseSettings, SettingsConfigDict

# Embedding dimensionality per linguistics backend — mirrors the
# fontem-linguistics catalog (src/domain/catalog.py). Used to keep the
# configured default backend consistent with the Neo4j vector index
# dims (see src/consolidator/neo4j/migrations.py); a backend whose dim
# differs from the index produces vectors the index silently refuses
# to hold, which kills embedding-cosine matching end to end.
EMBEDDING_BACKEND_DIMS: dict[str, int] = {
    "labse-local": 768,
    "minilm-local": 384,
    "mistral-embed": 1024,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONSOLIDATOR_", extra="ignore")

    neo4j_uri: str = "bolt://neo4j.fontem-prod.svc.cluster.local:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # Start flag-only; flip after the decision log has been observed.
    auto_merge_enabled: bool = False

    # Jaro-Winkler similarity on normalised names (uppercase + legal-form
    # suffixes stripped). 0.92 keeps very-close variants while rejecting
    # parent/subsidiary pairs like "SOCOTEC" vs "SOCOTEC CONSTRUCTION" (~0.88).
    fuzzy_name_threshold: float = 0.92
    gds_similarity_threshold: float = 0.7
    gds_top_k: int = 5

    # fontem-linguistics — translation + embedding service. Deployed as a
    # singleton in the linguistics-service namespace; every environment
    # (including prod) points at the same URL.
    linguistics_url: str = "http://fontem-linguistics.linguistics-service.svc.cluster.local:8080"
    linguistics_enabled: bool = True
    linguistics_timeout_s: float = 60.0
    linguistics_translation_backend: str = "nllb-local"
    # Default embedding backend MUST form a working pipeline with the
    # authority_name_embedding_idx vector index (768-d, migrations.py):
    # labse-local is 768-d, so enrichment writes vectors the index can
    # hold and embedding_cosine_authority can compare. mistral-embed
    # (1024-d) stays available via CONSOLIDATOR_LINGUISTICS_EMBEDDING_BACKEND,
    # but switching requires a matching-dim vector index — 1024-d vectors
    # are silently NOT indexed by a 768-d index, which is exactly how the
    # embedding-similarity feature shipped dead-on-arrival (zero
    # authorities enriched in prod). See EMBEDDING_BACKEND_DIMS above and
    # tests/unit/test_config.py which pins this consistency.
    linguistics_embedding_backend: str = "labse-local"

    # embedding_cosine_authority rule — flags Authority duplicates whose
    # LaBSE name-embedding cosine is above threshold. Never auto-merges.
    #
    # Calibration history:
    # - First canary (threshold=0.75, top_k=10) over-matched: LaBSE clusters
    #   by institutional role, so "music academy in X" ≈ "music academy in Y"
    #   at ~0.89. Role-lookalike pairs dominated the mid band.
    # - Second pass (current): 0.90 threshold cuts ~75% of the noise; top_k
    #   3 prevents quota-saturation per entity; a minimum Jaro-Winkler on the
    #   raw names forces some token overlap, rejecting pure role matches.
    #   cross-country-only because same-country string duplicates are already
    #   caught by fuzzy_name_same_country_authority.
    embedding_cosine_enabled: bool = True
    embedding_cosine_threshold: float = 0.90
    embedding_cosine_top_k: int = 3
    embedding_cosine_jaro_winkler_min: float = 0.45
    embedding_cosine_cross_country_only: bool = True
    # Homogeneity is enforced query-side: the Cypher WHERE gate on
    # node.name_embedding_encoder = $enc guarantees every compared pair
    # shares the same encoder. We no longer maintain an app-level
    # allowlist — any encoder present in the graph is legitimate for its
    # own siblings, and adding a new encoder (Mistral, MiniLM, ...) no
    # longer needs a config-and-redeploy dance.
    #
    # DEPRECATED — retained for backward-compat only, ignored by the rule.
    # (Was: 'labse@1.0.0-836121a'. When we ran on that whitelist while the
    # linguistics service was actually returning mistral-embed encoder-ids,
    # the rule silently abstained on every Authority for 52 days.)
    embedding_cosine_accepted_encoders: str = ""


settings = Settings()
