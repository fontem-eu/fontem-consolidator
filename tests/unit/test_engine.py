"""Engine unit test with mocked rules + mocked Neo4j writes."""

from unittest.mock import AsyncMock, patch

import pytest

from src.consolidator import engine
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule


class _FakeRule(Rule):
    name = "fake"
    entity_types = {"Company"}
    confidence = 0.9
    action = "flag"

    async def applies(self, entity):
        return True

    async def find_candidates(self, entity):
        return [
            Candidate(entity=Entity("Company", "gmr-B", {"name": "B"}), context={})
        ]

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
async def test_engine_records_run_and_decision():
    fake = _FakeRule()
    with patch("src.consolidator.engine.list_rules", return_value=[fake]), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {"name": "A"})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-1")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ) as rec, patch(
        "src.consolidator.engine.actions.execute", AsyncMock(return_value="flag")
    ):
        result = await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="gmr-A"
        )

    assert result.run_id == "run-1"
    assert result.rules_fired == 1
    assert len(result.decisions) == 1
    assert result.decisions[0]["outcome"] == "flag"
    rec.assert_awaited_once()


class _ConflictRule(Rule):
    """Higher-confidence rule that produces a conflict for (gmr-A, gmr-B)."""

    name = "conflict_rule"
    entity_types = {"Company"}
    confidence = 0.95
    action = "flag"

    async def applies(self, entity):
        return True

    async def find_candidates(self, entity):
        return [Candidate(entity=Entity("Company", "gmr-B", {"name": "B"}), context={})]

    async def resolve(self, entity, candidate):
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=0.95,
            entity_type="Company",
            details={"conflict": True},
        )


@pytest.mark.asyncio
async def test_engine_lets_multiple_flag_rules_fire_on_same_pair():
    """Engine no longer short-circuits on flag/conflict outcomes — both
    rules fire so each can append a detection to r.detections. The
    SAME_AS edge writer (_flag_same_as) makes r.conflict sticky-true so
    the higher-confidence rule's conflict signal isn't undone."""
    conflict = _ConflictRule()
    later_fuzzy = _FakeRule()
    with patch(
        "src.consolidator.engine.list_rules", return_value=[conflict, later_fuzzy]
    ), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {"name": "A"})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-1")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ), patch(
        "src.consolidator.engine.actions.execute",
        AsyncMock(side_effect=["conflict", "flag"]),
    ) as exec_mock:
        result = await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="gmr-A"
        )

    # BOTH rules executed — flag/conflict no longer short-circuit.
    assert exec_mock.await_count == 2
    outcomes = [d["outcome"] for d in result.decisions]
    assert outcomes == ["conflict", "flag"]


@pytest.mark.asyncio
async def test_engine_still_short_circuits_after_auto_merge():
    """auto_merge collapses the target node — running another rule on
    the same target afterward is undefined behaviour, so the engine
    must still short-circuit."""
    class _MergeRule(_FakeRule):
        action = "merge"
    merge = _MergeRule()
    later_fuzzy = _FakeRule()
    with patch(
        "src.consolidator.engine.list_rules", return_value=[merge, later_fuzzy]
    ), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {"name": "A"})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-2")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ), patch(
        "src.consolidator.engine.actions.execute",
        AsyncMock(side_effect=["auto_merge", "flag"]),
    ) as exec_mock:
        await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="gmr-A"
        )

    # Only the merge fired — fuzzy was suppressed because the target
    # node is gone.
    assert exec_mock.await_count == 1


@pytest.mark.asyncio
async def test_engine_handles_missing_entity():
    with patch("src.consolidator.engine.entities.load", AsyncMock(return_value=None)), patch(
        "src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-X")
    ), patch("src.consolidator.engine.audit.end_run", AsyncMock()) as end:
        result = await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="missing"
        )
    assert result.rules_fired == 0
    assert result.decisions == []
    end.assert_awaited_once()
