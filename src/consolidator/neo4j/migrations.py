from loguru import logger
from neo4j import AsyncDriver
from neo4j.exceptions import ClientError

# Dimensionality of the authority_name_embedding_idx vector index.
# MUST match the dim of the embedding backend the enrichment rule is
# configured with (settings.linguistics_embedding_backend →
# src.config.EMBEDDING_BACKEND_DIMS): Neo4j silently skips indexing any
# vector whose length differs from the index declaration, so a mismatch
# doesn't error — it just makes every k-NN lookup return nothing.
# Pinned by tests/unit/test_config.py.
AUTHORITY_NAME_EMBEDDING_DIMS = 1024

INDEX_CYPHER = [
    (
        "CREATE INDEX decisionlog_entity IF NOT EXISTS "
        "FOR (d:DecisionLog) ON (d.entity_type, d.source_id)"
    ),
    "CREATE INDEX decisionlog_decided_at IF NOT EXISTS FOR (d:DecisionLog) ON (d.decided_at)",
    "CREATE INDEX decisionlog_rule IF NOT EXISTS FOR (d:DecisionLog) ON (d.rule_name)",
    # audit.record_decision and audit.finish_run both look a run up by
    # run_id, and record_decision runs once per decision — dozens of
    # times per consolidated entity. Without this index that MATCH is a
    # NodeByLabelScan: measured in prod at 4,775,388 nodes scanned,
    # 9.55M db hits and 2.9s for a single-node lookup. At ~44 decisions
    # per entity that is roughly two minutes of pure scanning per
    # event, which is what capped the trigger at 0.06 events/sec.
    #
    # It also degrades without bound. Every consolidation writes another
    # ConsolidationRun node, so the scan this index removes was getting
    # slower with every run — the pipeline was throttling itself.
    #
    # Range index rather than a uniqueness constraint: run_id is a
    # uuid4 and should be unique, but asserting that over 4.7M existing
    # rows can fail the migration on legacy duplicates, and the point
    # lookup is what actually matters here.
    (
        "CREATE INDEX consolidationrun_run_id IF NOT EXISTS "
        "FOR (r:ConsolidationRun) ON (r.run_id)"
    ),
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
    # last_consolidated_at: the re-consolidation sweeper's rotation
    # cursor. The sweeper pages the stalest entities with
    #   WHERE n.last_consolidated_at IS NOT NULL
    #   ORDER BY n.last_consolidated_at ASC LIMIT $page
    # which the planner answers by walking this index in order. (Until
    # 2026-09-19 it ordered by coalesce(..., 1970) instead, which no index
    # can serve: every page was a full label scan plus a sort despite
    # this index. Never-swept entities, absent from the index, are found
    # by a separate, less frequent scan: sweeper._page_unstamped.)
    "CREATE INDEX company_last_consolidated IF NOT EXISTS "
    "FOR (c:Company) ON (c.last_consolidated_at)",
    "CREATE INDEX authority_last_consolidated IF NOT EXISTS "
    "FOR (a:Authority) ON (a.last_consolidated_at)",
    # lei / cik / vat: the three hard-identifier dedup rules
    # (_ExactIdRule subclasses) each run
    #     MATCH (c:Company) WHERE c.<id> = $value AND c.gmr_id <> $self_id
    # for every entity that carries the identifier. Unindexed, each of
    # those is a NodeByLabelScan over the whole Company graph —
    # 3,516,650 nodes in shared — and a company with all three pays it
    # three times.
    #
    # These are the auto-merge rules (force_auto_merge = True), so they
    # are also the ones that must run on every entity rather than being
    # skippable. Measured on shared 2026-09-07: Neo4j pinned at 3,884m
    # of a 4-core limit while the sweeper managed 6 companies/sec and
    # the trigger 0.72 events/sec — a full sweep of the Company graph
    # would have taken ~6.7 days, and the trigger's 7.07M-event backlog
    # ~113 days.
    #
    # Same failure shape as consolidationrun_run_id above, which is
    # what capped the trigger at 0.06 events/sec before it was added.
    # Every property these rules match on was already indexed except
    # the hard identifiers.
    "CREATE INDEX company_lei IF NOT EXISTS FOR (c:Company) ON (c.lei)",
    "CREATE INDEX company_cik IF NOT EXISTS FOR (c:Company) ON (c.cik)",
    "CREATE INDEX company_vat IF NOT EXISTS FOR (c:Company) ON (c.vat)",
    # registered_as + country: the national business-register ID
    # (GLEIF RegistrationAuthorityEntityID), matched only alongside an
    # agreeing country because the number is unique per jurisdiction,
    # not globally. Composite so the resolver's registered_as hard tier
    # is an index seek, not a Company label scan.
    "CREATE INDEX company_registered_as_country IF NOT EXISTS "
    "FOR (c:Company) ON (c.registered_as, c.country)",
    # national_id + country: the Authority's published legal id (eForms
    # buyer cbc:CompanyID / TED nationalRegistrationNumber). GET
    # /fiscal-id asks for it per hypothesis -- up to a dozen countries
    # for a bare 9-digit number -- so it must be a seek, not a label
    # scan per call. Composite for the same reason as registered_as.
    "CREATE INDEX authority_national_id_country IF NOT EXISTS "
    "FOR (a:Authority) ON (a.national_id, a.country)",
    # Vector index on Authority name_embedding (mistral-embed, 1024-d,
    # cosine — see AUTHORITY_NAME_EMBEDDING_DIMS).
    # Powers the embedding_cosine_authority rule's k-NN lookup; without
    # it the rule would fall back to a full 61k × 768 dot-product scan
    # per candidate and become unusable. Only one encoder is loaded at
    # a time so a single index is sufficient — if we ever run two
    # encoders in parallel we add a second index keyed by encoder_id.
    # Fulltext index on Company.name — the fuzzy_name_same_country rule
    # retrieves candidates via db.index.fulltext.queryNodes('company_name_ft').
    # Without it that rule silently returns no candidates (it catches the
    # "no such index" error), so fuzzy company dedup never runs. Missing in
    # prod until 2026-07-23; the sweeper surfaced it.
    "CREATE FULLTEXT INDEX company_name_ft IF NOT EXISTS FOR (n:Company) ON EACH [n.name]",
    (
        "CREATE VECTOR INDEX authority_name_embedding_idx IF NOT EXISTS "
        "FOR (a:Authority) ON (a.name_embedding) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {AUTHORITY_NAME_EMBEDDING_DIMS}, "
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
    # Sweep up SAME_AS self-loops left behind by merges that ran before
    # produceSelfRel:false was set on apoc.refactor.mergeNodes. Merging a
    # duplicate into its canonical turned the SAME_AS edge BETWEEN the pair
    # into a self-loop on the survivor, and 571 had accumulated by
    # 2026-09-02 — the standing refs.sameas_no_selfloop block-tier failure.
    #
    # Safe to delete outright rather than review: the edge asserts a node is
    # the same entity as itself, which is true but carries no information,
    # and it corrupts every consumer that walks SAME_AS to build clusters.
    # Nothing is lost — the merge it came from is already recorded as a
    # :MergeEvent.
    #
    # Idempotent and cheap once drained: with the flag set, no new ones
    # appear and this matches nothing on subsequent runs.
    #
    # Covers :SAME_AS_CANDIDATE too. A self-referential proposal is the
    # same rule bug wearing the other relationship type, and after the
    # proposal/assertion split almost everything is a proposal — sweeping
    # only assertions would clean the half that barely fills up.
    """
    MATCH (a)-[r:SAME_AS|SAME_AS_CANDIDATE]->(a)
    CALL (r) {
        DELETE r
    } IN TRANSACTIONS OF 1000 ROWS
    """,
]


async def _drop_stale_vector_index(session) -> None:
    """CREATE VECTOR INDEX IF NOT EXISTS never updates an existing index,
    so a dims change (768 -> 1024 when the encoder moved from labse to
    mistral-embed) would silently keep the old index and the new vectors
    would never be indexed. Drop it iff its dimensions disagree."""
    result = await session.run(
        "SHOW VECTOR INDEXES YIELD name, options "
        "WHERE name = 'authority_name_embedding_idx' "
        "RETURN options.indexConfig['vector.dimensions'] AS dims",
    )
    record = await result.single()
    if record is not None and record["dims"] != AUTHORITY_NAME_EMBEDDING_DIMS:
        await session.run("DROP INDEX authority_name_embedding_idx")


#: What CREATE ... IF NOT EXISTS raises when a concurrent session created
#: the same schema rule first: the existence check is not atomic with the
#: create. Both consolidator replicas and the sweeper run apply() at
#: startup, so when a node restart brings them up together one of them
#: used to lose the race and exit 1 (seen on prod 2026-09-10, company_cik).
#: The rule it asked for exists, which is the outcome it wanted.
_CREATED_CONCURRENTLY = frozenset({
    "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
    "Neo.ClientError.Schema.IndexAlreadyExists",
    "Neo.ClientError.Schema.ConstraintAlreadyExists",
})


async def _ensure(driver: AsyncDriver, database: str, stmt: str) -> None:
    # One session per statement, so a statement that lost the race leaves
    # nothing behind for the next one to trip over.
    async with driver.session(database=database) as session:
        try:
            await session.run(stmt)
        except ClientError as exc:
            if exc.code not in _CREATED_CONCURRENTLY:
                raise
            logger.info(
                "consolidator: created concurrently by another starter: {}",
                stmt.split(" IF NOT EXISTS", 1)[0],
            )


# Backfill failures that mean "the graph was busy", not "the statement
# is wrong": worth skipping and retrying on the next boot.
_TRANSIENT_BACKFILL_ERRORS = frozenset({
    "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
    "Neo.ClientError.Transaction.LockClientStopped",
    "Neo.TransientError.Transaction.LockAcquisitionTimeout",
})


async def apply(driver: AsyncDriver, database: str) -> None:
    async with driver.session(database=database) as session:
        await _drop_stale_vector_index(session)
    for stmt in INDEX_CYPHER:
        await _ensure(driver, database, stmt)
    logger.info(
        "consolidator: neo4j indexes ensured ({} statements)",
        len(INDEX_CYPHER),
    )
    # Backfill runs implicit-tx style (CALL { } IN TRANSACTIONS
    # requires it), so each statement uses its own session that
    # the driver auto-commits.
    #
    # A timeout here must NOT take the process down. Each statement is a
    # full label scan whose cost grows with the graph, and it runs on
    # every boot — so on a cold Neo4j (page cache empty after a cluster
    # restart) it reliably exceeds the server's transaction timeout. On
    # 2026-09-25 that put the prod sweeper in CrashLoopBackOff with 9
    # restarts: it had already consolidated hundreds of entities each
    # time before the backfill killed it.
    #
    # Skipping is safe. name_clean is best-effort here: the sink writes
    # it on every :Company / :Authority upsert (apoc.text.clean in
    # _merge_company_rows), so this only catches rows written before
    # that existed. Missing it costs the exact-name rule a match on
    # those rows — a degradation the next boot retries — where dying
    # costs every rule on every entity.
    completed = 0
    for stmt in BACKFILL_CYPHER:
        try:
            async with driver.session(database=database) as session:
                await session.run(stmt)
            completed += 1
        except ClientError as exc:
            if exc.code not in _TRANSIENT_BACKFILL_ERRORS:
                raise
            logger.warning(
                "consolidator: name_clean backfill statement timed out "
                "({}); continuing without it — the sweep matters more "
                "than the backfill",
                exc.code,
            )
    logger.info(
        "consolidator: name_clean backfill complete ({}/{} statements)",
        completed, len(BACKFILL_CYPHER),
    )
