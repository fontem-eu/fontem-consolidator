"""exclude_rule_prefix: skip rules whose name starts with the given prefix.
Used by the full-scan script to bypass GDS rules (which reproject the
whole subgraph per call)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.consolidator import engine
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class _Always(Rule):
    entity_types = {"Company"}
    confidence = 0.9
    action = "flag"

    def __init__(self, name):
        self.name = name

    async def applies(self, entity):
        return True

    async def find_candidates(self, entity):
        return [Candidate(entity=Entity("Company", "gmr-B", {}), context={})]

    async def resolve(self, entity, candidate):
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=0.9,
            entity_type="Company",
            details={},
        )


@pytest.mark.asyncio
async def test_exclude_rule_prefix_skips_matching_rules():
    rules = [_Always("gds_node_similarity_company"), _Always("exact_lei_match")]
    with patch("src.consolidator.engine.list_rules", return_value=rules), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-1")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ), patch(
        "src.consolidator.engine.actions.execute", AsyncMock(return_value="flag")
    ):
        result = await engine.consolidate(
            AsyncMock(),
            "neo4j",
            entity_type="Company",
            entity_id="gmr-A",
            exclude_rule_prefix="gds_",
        )
    # Only exact_lei_match fired; gds_* skipped
    assert result.rules_fired == 1
    assert result.decisions[0]["rule_name"] == "exact_lei_match"


@pytest.mark.asyncio
async def test_no_prefix_runs_all_rules():
    rules = [_Always("gds_node_similarity_company"), _Always("exact_lei_match")]
    with patch("src.consolidator.engine.list_rules", return_value=rules), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-2")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ), patch(
        "src.consolidator.engine.actions.execute", AsyncMock(return_value="flag")
    ):
        result = await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="gmr-A"
        )
    # Both rules evaluated AND both actions fired — flag no longer
    # short-circuits, each rule appends a detection to r.detections so
    # the reviewer sees the full evidence on the SAME_AS edge. Only
    # graph-mutating outcomes (auto_merge / auto_link) still short-circuit.
    assert result.rules_fired == 2
    assert len(result.decisions) == 2
    assert {d["rule_name"] for d in result.decisions} == {
        "gds_node_similarity_company", "exact_lei_match",
    }
