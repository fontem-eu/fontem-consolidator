from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONSOLIDATOR_", extra="ignore")

    neo4j_uri: str = "bolt://neo4j.gmr.svc.cluster.local:7687"
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

    # gmr-linguistics — translation + embedding service. Deployed only in
    # prod (singleton); non-prod consolidators point at the same URL.
    linguistics_url: str = "http://fontem-linguistics.linguistics-service.svc.cluster.local:8080"
    linguistics_enabled: bool = True
    linguistics_timeout_s: float = 60.0
    linguistics_translation_backend: str = "mistral"
    linguistics_embedding_backend: str = "mistral-embed"

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
    # Comma-separated list of encoder-ids whose vectors are safe to
    # compare. Cross-encoder cosines are meaningless; this is the guard.
    # If a row's encoder_id isn't in this list the rule abstains.
    embedding_cosine_accepted_encoders: str = "labse@1.0.0-836121a"


settings = Settings()
