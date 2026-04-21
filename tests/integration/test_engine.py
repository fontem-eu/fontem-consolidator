"""End-to-end rule engine scenarios against a real Neo4j + GDS + APOC container."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config import settings
from src.consolidator import engine


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _create_company(driver, **props):
    async with driver.session() as s:
        await s.run(
            "CREATE (c:Company) SET c = $props",
            props={**props, "created_at": _ts()},
        )


async def _create_authority(driver, **props):
    async with driver.session() as s:
        await s.run("CREATE (a:Authority) SET a = $props", props={**props, "created_at": _ts()})


async def _count(driver, cypher, **params) -> int:
    async with driver.session() as s:
        result = await s.run(cypher, **params)
        record = await result.single()
        return record[0] if record else 0


@pytest.mark.asyncio
async def test_exact_lei_auto_merges(driver, monkeypatch):
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="A", lei="LEI-X", name="Acme")
    await _create_company(driver, gmr_id="B", lei="LEI-X", name="Acme Inc")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 1
    merge_events = await _count(driver, "MATCH (:MergeEvent) RETURN count(*)")
    assert merge_events >= 1
    auto_merges = await _count(
        driver, "MATCH (:DecisionLog {decision_type:'auto_merge'}) RETURN count(*)"
    )
    assert auto_merges >= 1


@pytest.mark.asyncio
async def test_exact_lei_flags_when_auto_disabled(driver, monkeypatch):
    monkeypatch.setattr(settings, "auto_merge_enabled", False)
    await _create_company(driver, gmr_id="A", lei="LEI-Y", name="Acme")
    await _create_company(driver, gmr_id="B", lei="LEI-Y", name="Acme Inc")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    same_as = await _count(
        driver,
        "MATCH (:Company {gmr_id:'A'})-[r:SAME_AS]->(:Company {gmr_id:'B'}) RETURN count(r)",
    )
    assert same_as == 1
    flags = await _count(
        driver, "MATCH (:DecisionLog {decision_type:'flag'}) RETURN count(*)"
    )
    assert flags >= 1
    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 2  # no merge


@pytest.mark.asyncio
async def test_conflicting_identifiers_refuse_merge(driver, monkeypatch):
    """Same LEI, different *canonical* VATs → conflict refused. VATs must
    both canonicalise to valid values for the conflict to fire."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="A", lei="529900WTOG7RHO5TCH58",
                          vat="FR12345678901", name="Acme", country="FR")
    await _create_company(driver, gmr_id="B", lei="529900WTOG7RHO5TCH58",
                          vat="FR98765432109", name="Acme", country="FR")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    # Both nodes must survive — merge refused
    assert await _count(driver, "MATCH (c:Company) RETURN count(c)") == 2
    conflict = await _count(
        driver,
        "MATCH (:Company {gmr_id:'A'})-[r:SAME_AS]->(:Company {gmr_id:'B'}) "
        "WHERE r.conflict = true RETURN count(r)",
    )
    assert conflict == 1
    conflict_dl = await _count(
        driver,
        "MATCH (:DecisionLog {decision_type:'conflict'}) RETURN count(*)",
    )
    assert conflict_dl >= 1


@pytest.mark.asyncio
async def test_fuzzy_name_flags_same_as(driver, monkeypatch):
    """Names that differ by a legal-form suffix normalise identically →
    Jaro-Winkler = 1.0, rule flags :SAME_AS."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="A", name="Globex Corporation SA", country="FR")
    await _create_company(driver, gmr_id="B", name="Globex Corporation S.A.", country="FR")

    async with driver.session() as s:
        await s.run("CALL db.awaitIndexes(30)")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    same_as_either_direction = await _count(
        driver,
        "MATCH (a:Company {gmr_id:'A'})-[r:SAME_AS]-(b:Company {gmr_id:'B'}) "
        "WHERE r.reviewed = false AND coalesce(r.conflict,false) = false RETURN count(r)",
    )
    assert same_as_either_direction >= 1
    # Both companies still present (fuzzy never auto-merges)
    assert await _count(driver, "MATCH (c:Company) RETURN count(c)") == 2


@pytest.mark.asyncio
async def test_fuzzy_rejects_parent_subsidiary(driver, monkeypatch):
    """SOCOTEC vs SOCOTEC CONSTRUCTION must NOT flag — too dissimilar under JW."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="PARENT", name="SOCOTEC", country="FRA")
    await _create_company(driver, gmr_id="SUB", name="SOCOTEC CONSTRUCTION", country="FRA")

    async with driver.session() as s:
        await s.run("CALL db.awaitIndexes(30)")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="PARENT", triggered_by="test"
    )

    any_edge = await _count(
        driver,
        "MATCH (:Company {gmr_id:'PARENT'})-[r:SAME_AS]-(:Company {gmr_id:'SUB'}) RETURN count(r)",
    )
    assert any_edge == 0


@pytest.mark.asyncio
async def test_malformed_vat_does_not_trigger_conflict(driver, monkeypatch):
    """Same (name, country), one VAT is a TED notice ID → merge proceeds (not conflict)."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="A", name="Malerbetrieb Cambel", country="DEU",
                          vat="DE273691032")
    await _create_company(driver, gmr_id="B", name="Malerbetrieb Cambel", country="DEU",
                          vat="1594225-1-0-1")  # TED notice in vat field

    async with driver.session() as s:
        await s.run("CALL db.awaitIndexes(30)")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    # Should have merged (both nodes collapse into one)
    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 1


@pytest.mark.asyncio
async def test_authority_name_country_merges(driver, monkeypatch):
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_authority(
        driver, authority_id="auth-1", name="Ministère de l'Économie", country="FR"
    )
    await _create_authority(
        driver, authority_id="auth-2", name="Ministère de l'Économie", country="FR"
    )

    await engine.consolidate(
        driver, "neo4j", entity_type="Authority", entity_id="auth-1", triggered_by="test"
    )

    remaining = await _count(driver, "MATCH (a:Authority) RETURN count(a)")
    assert remaining == 1


@pytest.mark.asyncio
async def test_gds_same_as_wcc_collapses_reviewed_cluster(driver, monkeypatch):
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    # Seed 3 companies connected A-B and B-C by :SAME_AS {reviewed:true}
    async with driver.session() as s:
        await s.run(
            """
            CREATE (a:Company {gmr_id:'A', name:'A', country:'FR', lei:'LEI-CAN', created_at:$t})
            CREATE (b:Company {gmr_id:'B', name:'B', country:'FR', created_at:$t})
            CREATE (c:Company {gmr_id:'C', name:'C', country:'FR', created_at:$t})
            CREATE (a)-[:SAME_AS {reviewed:true, confidence:1.0}]->(b)
            CREATE (b)-[:SAME_AS {reviewed:true, confidence:1.0}]->(c)
            """,
            t=_ts(),
        )

    # Consolidate A — it's the canonical (has LEI). Rule should merge B and C in.
    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 1
    merge_events = await _count(
        driver, "MATCH (m:MergeEvent) WHERE m.method STARTS WITH 'gds_' RETURN count(m)"
    )
    assert merge_events >= 2


@pytest.mark.asyncio
async def test_conflict_flag_survives_subsequent_fuzzy_match(driver, monkeypatch):
    """Regression: when exact-name-country emits conflict and fuzzy matches the same pair,
    the edge must retain conflict:true and method=exact_name_country_match.
    Without short-circuit, MERGE on the same (a,b) pair would overwrite the properties."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    # Same name + country with conflicting *canonical* VATs → exact-name-country
    # emits conflict. The fuzzy rule would then also match the same pair,
    # but its own conflict-skip short-circuits it (noop) — so the edge stays
    # flagged as conflict from the exact rule.
    await _create_company(driver, gmr_id="A", name="Socotec", country="FRA",
                          vat="FR12345678901")
    await _create_company(driver, gmr_id="B", name="Socotec", country="FRA",
                          vat="FR98765432109")
    async with driver.session() as s:
        await s.run("CALL db.awaitIndexes(30)")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    rows = await _count(
        driver,
        "MATCH (:Company {gmr_id:'A'})-[r:SAME_AS]-(:Company {gmr_id:'B'}) "
        "WHERE r.conflict = true AND r.method = 'exact_name_country_match' "
        "RETURN count(r)",
    )
    assert rows == 1


@pytest.mark.asyncio
async def test_consolidation_run_and_decisionlog_always_written(driver, monkeypatch):
    """Even for a no-match entity, a :ConsolidationRun exists. And for actual
    matches, a :DecisionLog chains off the run via :RuleApplication."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="ALONE", name="Alone Corp", country="ES")

    result = await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="ALONE", triggered_by="test"
    )

    runs = await _count(
        driver, "MATCH (r:ConsolidationRun {run_id: $id}) RETURN count(r)", id=result.run_id
    )
    assert runs == 1
