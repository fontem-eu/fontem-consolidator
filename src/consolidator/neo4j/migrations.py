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
]


async def apply(driver: AsyncDriver, database: str) -> None:
    async with driver.session(database=database) as session:
        for stmt in INDEX_CYPHER:
            await session.run(stmt)
    logger.info("consolidator: neo4j indexes ensured ({} statements)", len(INDEX_CYPHER))
