from loguru import logger
from neo4j import AsyncDriver

INDEX_CYPHER = [
    "CREATE INDEX decisionlog_entity IF NOT EXISTS FOR (d:DecisionLog) ON (d.entity_type, d.source_id)",
    "CREATE INDEX decisionlog_decided_at IF NOT EXISTS FOR (d:DecisionLog) ON (d.decided_at)",
    "CREATE INDEX decisionlog_rule IF NOT EXISTS FOR (d:DecisionLog) ON (d.rule_name)",
    "CREATE INDEX consolidationrun_started IF NOT EXISTS FOR (r:ConsolidationRun) ON (r.started_at)",
    # historic_leis: retired/lapsed LEIs that previously identified this entity.
    # Index lets us look up a company from any of its past LEIs when
    # ETL writes come in referencing an old identifier.
    "CREATE INDEX company_historic_leis IF NOT EXISTS FOR (c:Company) ON (c.historic_leis)",
    # Vector index on Authority name_embedding (LaBSE, 768-d, cosine).
    # Powers the embedding_cosine_authority rule's k-NN lookup; without
    # it the rule would fall back to a full 61k × 768 dot-product scan
    # per candidate and become unusable. Only one encoder is loaded at
    # a time so a single index is sufficient — if we ever run two
    # encoders in parallel we add a second index keyed by encoder_id.
    (
        "CREATE VECTOR INDEX authority_name_embedding_idx IF NOT EXISTS "
        "FOR (a:Authority) ON (a.name_embedding) "
        "OPTIONS {indexConfig: {"
        "`vector.dimensions`: 768, "
        "`vector.similarity_function`: 'cosine'"
        "}}"
    ),
]


async def apply(driver: AsyncDriver, database: str) -> None:
    async with driver.session(database=database) as session:
        for stmt in INDEX_CYPHER:
            await session.run(stmt)
    logger.info("consolidator: neo4j indexes ensured ({} statements)", len(INDEX_CYPHER))
