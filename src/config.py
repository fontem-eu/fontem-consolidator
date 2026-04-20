from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONSOLIDATOR_", extra="ignore")

    neo4j_uri: str = "bolt://neo4j.gmr.svc.cluster.local:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # Start flag-only; flip after the decision log has been observed.
    auto_merge_enabled: bool = False

    fuzzy_name_threshold: float = 0.85
    gds_similarity_threshold: float = 0.7
    gds_top_k: int = 5


settings = Settings()
