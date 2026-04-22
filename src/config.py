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
    linguistics_url: str = "http://gmr-linguistics.gmr.svc.cluster.local:8080"
    linguistics_enabled: bool = True
    linguistics_timeout_s: float = 60.0
    linguistics_translation_backend: str = "mistral"
    linguistics_embedding_backend: str = "mistral-embed"


settings = Settings()
