from loguru import logger
from neo4j import AsyncDriver

INDEX_CYPHER = [
    (
        "CREATE INDEX decisionlog_entity IF NOT EXISTS "
        "FOR (d:DecisionLog) ON (d.entity_type, d.source_id)"
    ),
    "CREATE INDEX decisionlog_decided_at IF NOT EXISTS FOR (d:DecisionLog) ON (d.decided_at)",
    "CREATE INDEX decisionlog_rule IF NOT EXISTS FOR (d:DecisionLog) ON (d.rule_name)",
    (
        "CREATE INDEX consolidationrun_started IF NOT EXISTS "
        "FOR (r:ConsolidationRun) ON (r.started_at)"
    ),
    # Composite index used by the resume-aware sweep's pending-IDs query
    # ("entities with no recent ConsolidationRun"). Without this the
    # query is a label scan and trips Neo4j's 30s tx timeout once the
    # ConsolidationRun history grows past a few thousand rows.
    (
        "CREATE INDEX consolidationrun_entity IF NOT EXISTS "
        "FOR (r:ConsolidationRun) "
        "ON (r.entity_type, r.entity_id, r.started_at)"
    ),
    # gmr_id: primary lookup key for every Company MERGE that the
    # neo4j-sink performs (per-event path: one MERGE per UpsertEntity).
    # Without this index a 3.3M-row Company graph turns each MERGE into
    # a ~237k-DbHits label scan; the GLEIF replay collapsed to ~10/s.
    # With the index, MERGE is a sub-ms index seek and the same replay
    # runs at the sink's batch-write ceiling.
    "CREATE INDEX company_gmr_id IF NOT EXISTS FOR (c:Company) ON (c.gmr_id)",
    # ted_notice_id: same problem as company_gmr_id but for Contract.
    # Every UpsertContract event from the TED loader MERGEs by
    # ted_notice_id. With ~100k Contract nodes per monthly TED package
    # the second-month run drops to ~20/s without this index because
    # each MERGE becomes a label scan that grows with the existing
    # Contract count. Added live in both staging + prod Neo4j on
    # 2026-05-27; this migration ensures fresh deploys carry it.
    "CREATE INDEX contract_ted_notice_id IF NOT EXISTS FOR (c:Contract) ON (c.ted_notice_id)",
    # historic_leis: retired/lapsed LEIs that previously identified this entity.
    # Index lets us look up a company from any of its past LEIs when
    # ETL writes come in referencing an old identifier.
    "CREATE INDEX company_historic_leis IF NOT EXISTS FOR (c:Company) ON (c.historic_leis)",
    # name_clean: pre-computed apoc.text.clean(name), materialised by
    # the Neo4j sink at projection time. The resolver's Tier 3 +
    # the dedup rules used to do
    #   WHERE apoc.text.clean(c.name) = apoc.text.clean($name)
    # which bypassed any index on c.name (function on a property
    # forces a full scan; ~3.3M Company nodes ≈ 10s/query). After
    # moving the cleaned form into a property we range-index it;
    # the resolver query becomes a sub-100ms index lookup.
    "CREATE INDEX company_name_clean IF NOT EXISTS FOR (c:Company) ON (c.name_clean)",
    "CREATE INDEX authority_name_clean IF NOT EXISTS FOR (a:Authority) ON (a.name_clean)",
    # registered_as + country: the national business-register ID
    # (GLEIF RegistrationAuthorityEntityID), matched only alongside an
    # agreeing country because the number is unique per jurisdiction,
    # not globally. Composite so the resolver's registered_as hard tier
    # is an index seek, not a Company label scan.
    "CREATE INDEX company_registered_as_country IF NOT EXISTS "
    "FOR (c:Company) ON (c.registered_as, c.country)",
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


# One-shot backfill for ``name_clean`` on existing rows. Runs in
# small chunks to avoid a 3.3M-row tx that would blow the
# transaction memory cap. The CALL { ... } IN TRANSACTIONS form
# commits per chunk so a partial failure leaves the cluster in a
# valid state and the next pod restart picks up where we left off.
BACKFILL_CYPHER = [
    """
    MATCH (c:Company)
    WHERE c.name IS NOT NULL AND c.name_clean IS NULL
    CALL (c) {
        SET c.name_clean = apoc.text.clean(c.name)
    } IN TRANSACTIONS OF 5000 ROWS
    """,
    """
    MATCH (a:Authority)
    WHERE a.name IS NOT NULL AND a.name_clean IS NULL
    CALL (a) {
        SET a.name_clean = apoc.text.clean(a.name)
    } IN TRANSACTIONS OF 5000 ROWS
    """,
]


async def apply(driver: AsyncDriver, database: str) -> None:
    async with driver.session(database=database) as session:
        for stmt in INDEX_CYPHER:
            await session.run(stmt)
    logger.info(
        "consolidator: neo4j indexes ensured ({} statements)",
        len(INDEX_CYPHER),
    )
    # Backfill runs implicit-tx style (CALL { } IN TRANSACTIONS
    # requires it), so each statement uses its own session that
    # the driver auto-commits.
    for stmt in BACKFILL_CYPHER:
        async with driver.session(database=database) as session:
            await session.run(stmt)
    logger.info(
        "consolidator: name_clean backfill complete ({} statements)",
        len(BACKFILL_CYPHER),
    )
