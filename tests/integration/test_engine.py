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
async def test_exact_lei_force_auto_merges_even_when_gate_disabled(driver, monkeypatch):
    """ExactLeiMatch sets `force_auto_merge = True` — same LEI on two
    Company nodes is the canonical same-entity signal (LEIs are
    issued one-per-entity by definition). The decision must merge
    even when the global `auto_merge_enabled` gate is False, because
    routing this to the review queue would create unbounded review
    backlog for every GLEIF + EU-listings + US-companies overlap.
    """
    monkeypatch.setattr(settings, "auto_merge_enabled", False)
    await _create_company(driver, gmr_id="A", lei="LEI-Y", name="Acme")
    await _create_company(driver, gmr_id="B", lei="LEI-Y", name="Acme Inc")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 1  # merged despite gate=False
    auto_merges = await _count(
        driver, "MATCH (:DecisionLog {decision_type:'auto_merge'}) RETURN count(*)"
    )
    assert auto_merges >= 1


@pytest.mark.asyncio
async def test_fuzzy_match_still_flags_when_gate_disabled(driver, monkeypatch):
    """Fuzzy name matching (`FuzzyNameSameCountry`) is NOT a
    deterministic-identifier rule — it stays subject to the global
    gate. Verifies force_auto_merge is opt-in per rule, not a
    cross-the-board override.
    """
    monkeypatch.setattr(settings, "auto_merge_enabled", False)
    # Two companies with NO shared hard identifier, only similar
    # names + same country — exercises the fuzzy path, not the LEI
    # path.
    await _create_company(driver, gmr_id="C", name="ACME CORPORATION", country="US")
    await _create_company(driver, gmr_id="D", name="ACME CORP", country="US")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="C", triggered_by="test"
    )

    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 2  # NOT merged — fuzzy rules respect the global gate


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
    """Names close under Jaro-Winkler but NOT identical after apoc.text.clean
    (so they don't collapse via the exact rule) → fuzzy flag."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    # These differ by a locale-specific word; apoc.text.clean won't fold them,
    # but Jaro-Winkler will score them high.
    await _create_company(driver, gmr_id="A",
                          name="Globex Corporation International", country="FR")
    await _create_company(driver, gmr_id="B",
                          name="Globex Corporation Internacional", country="FR")

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
async def test_whitespace_variants_auto_merge_via_exact_rule(driver, monkeypatch):
    """Two companies whose names differ only by whitespace/punctuation are
    caught by apoc.text.clean in exact_name_country_match and auto-merged."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="A", name="NEURAXPHARM FRANCE (Rang 1)", country="FRA")
    await _create_company(driver, gmr_id="B", name="NEURAXPHARM FRANCE ( Rang 1)", country="FRA")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 1
    merges = await _count(
        driver, "MATCH (:MergeEvent {method:'exact_name_country_match'}) RETURN count(*)"
    )
    assert merges >= 1


@pytest.mark.asyncio
async def test_ukrainian_boilerplate_not_flagged(driver, monkeypatch):
    """Two Ukrainian LLCs with different distinctive names but shared LLC
    boilerplate must NOT produce a fuzzy flag — the boilerplate is stripped
    before Jaro-Winkler."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(driver, gmr_id="A", country="UKR",
                          name='ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ "АНСУ"')
    await _create_company(driver, gmr_id="B", country="UKR",
                          name='ТОВАРИСТВО З ОБМЕЖЕНОЮ ВІДПОВІДАЛЬНІСТЮ "АЕРОК"')
    async with driver.session() as s:
        await s.run("CALL db.awaitIndexes(30)")

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    any_edge = await _count(
        driver,
        "MATCH (:Company {gmr_id:'A'})-[r:SAME_AS]-(:Company {gmr_id:'B'}) RETURN count(r)",
    )
    assert any_edge == 0


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
async def test_successor_lei_merges_and_preserves_historic(driver, monkeypatch):
    """Active + inactive nodes with same name/country/LOU → merge into active,
    retired LEI lands on canonical.historic_leis."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(
        driver, gmr_id="ACTIVE",
        name="kleiner und bold GmbH", country="DEU",
        lei="529900ACTIVEXXXXXXX1", active=True,
    )
    await _create_company(
        driver, gmr_id="RETIRED",
        name="kleiner und bold GmbH", country="DEU",
        lei="529900RETIREDXXXXXX2", active=False,
    )

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="ACTIVE", triggered_by="test"
    )

    # Only the ACTIVE node survives
    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 1

    # historic_leis has the retired LEI
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Company {gmr_id:'ACTIVE'}) RETURN c.historic_leis AS h"
        )
        rec = await r.single()
    assert rec is not None
    assert "529900RETIREDXXXXXX2" in (rec["h"] or [])

    # MergeEvent carries the method + retired_lei
    async with driver.session() as s:
        r = await s.run(
            "MATCH (m:MergeEvent {method:'successor_lei_match'}) "
            "RETURN m.retired_lei AS retired"
        )
        rec = await r.single()
    assert rec is not None
    assert rec["retired"] == "529900RETIREDXXXXXX2"


@pytest.mark.asyncio
async def test_successor_rule_ignores_two_active_entities(driver, monkeypatch):
    """CAISSE / sibling case: two ACTIVE nodes sharing name+country+LOU must
    NOT be merged by the successor rule (that'd collapse legitimate siblings).
    The existing conflict rule fires instead."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_company(
        driver, gmr_id="A",
        name="CAISSE REGLEMENTS PECUNIAIRES AVOCATS", country="FRA",
        lei="969500SDDWXX8CRI7V10", active=True,
    )
    await _create_company(
        driver, gmr_id="B",
        name="CAISSE REGLEMENTS PECUNIAIRES AVOCATS", country="FRA",
        lei="969500BW0ZWO0WZGB598", active=True,
    )

    await engine.consolidate(
        driver, "neo4j", entity_type="Company", entity_id="A", triggered_by="test"
    )

    # Both siblings survive
    remaining = await _count(driver, "MATCH (c:Company) RETURN count(c)")
    assert remaining == 2
    # No successor merge happened
    succ_merges = await _count(
        driver, "MATCH (:MergeEvent {method:'successor_lei_match'}) RETURN count(*)"
    )
    assert succ_merges == 0
    # The exact_name_country_match rule correctly flagged this as conflict
    conflict = await _count(
        driver,
        "MATCH (:Company {gmr_id:'A'})-[r:SAME_AS]-(:Company {gmr_id:'B'}) "
        "WHERE r.conflict = true RETURN count(r)",
    )
    assert conflict == 1


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
async def test_authority_cross_country_flags_same_as(driver, monkeypatch):
    """EU-body pattern: same clean name across different countries → flag :SAME_AS.
    Must NOT auto-merge (human review required)."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_authority(driver, authority_id="eeas-bel",
                            name="European External Action Service (EEAS)", country="BEL")
    await _create_authority(driver, authority_id="eeas-mus",
                            name="European External Action Service (EEAS)", country="MUS")
    async with driver.session() as s:
        await s.run("CALL db.awaitIndexes(30)")

    await engine.consolidate(
        driver, "neo4j", entity_type="Authority", entity_id="eeas-bel", triggered_by="test"
    )

    # Both authorities still exist (flag, not merge)
    remaining = await _count(driver, "MATCH (a:Authority) RETURN count(a)")
    assert remaining == 2
    # SAME_AS edge with the new rule's method
    flag = await _count(
        driver,
        "MATCH (:Authority {authority_id:'eeas-bel'})-[r:SAME_AS]-"
        "(:Authority {authority_id:'eeas-mus'}) "
        "WHERE r.method = 'exact_name_any_country_authority' RETURN count(r)",
    )
    assert flag == 1


@pytest.mark.asyncio
async def test_authority_cross_country_skipped_when_same_country_merge_wins(
    driver, monkeypatch
):
    """Two authorities with same name AND same country must be auto-merged by
    the higher-confidence same-country rule; the cross-country rule must not
    then write a spurious SAME_AS on a non-existent node."""
    monkeypatch.setattr(settings, "auto_merge_enabled", True)
    await _create_authority(driver, authority_id="auth-a",
                            name="Ministère de l'Économie", country="FR")
    await _create_authority(driver, authority_id="auth-b",
                            name="Ministère de l'Économie", country="FR")

    await engine.consolidate(
        driver, "neo4j", entity_type="Authority", entity_id="auth-a", triggered_by="test"
    )

    remaining = await _count(driver, "MATCH (a:Authority) RETURN count(a)")
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
