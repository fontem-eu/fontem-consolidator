"""Tests that match-style outcomes emit an AssertSameAs event so
Virtuoso (and replay-from-zero of Neo4j) sees the equivalence.

Path-(a) future state: this becomes the *only* write — the Neo4j
direct write goes away once the Neo4j sink owns the detection-arrays
projection. For Phase D we keep the Neo4j-direct write and emit
the event alongside it.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator import actions
from src.consolidator.rules.base import Decision, Entity


def _decision(action: str, *, conflict: bool = False, rule: str = "rule_x") -> Decision:
    return Decision(
        rule_name=rule,
        action=action,
        source_id="A",
        target_id="B",
        confidence=0.93,
        entity_type="Company",
        details={"conflict": conflict} if conflict else {},
    )


@pytest.mark.asyncio
async def test_flag_emits_assert_same_as():
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_flag_same_as", new=AsyncMock()) as fsa, \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("flag"),
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "flag"
    fsa.assert_awaited_once()
    emit.assert_awaited_once()
    kwargs = emit.await_args.kwargs
    assert kwargs["a_iri"] == "http://data.fontem.eu/id/Company/A"
    assert kwargs["b_iri"] == "http://data.fontem.eu/id/Company/B"
    assert kwargs["confidence"] == pytest.approx(0.93)
    assert kwargs["method"] == "rule_x"


@pytest.mark.asyncio
async def test_conflict_emits_assert_same_as():
    """A conflict-flagged match still asserts equivalence — the
    review queue's decision lands separately on the SAME_AS edge."""
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_flag_same_as", new=AsyncMock()), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("flag", conflict=True),
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "conflict"
    emit.assert_awaited_once()


@pytest.mark.asyncio
async def test_merge_emits_assert_same_as_when_auto_merge_on():
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_merge", new=AsyncMock()), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit, \
         patch.object(actions.settings, "auto_merge_enabled", True):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("merge"),
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "auto_merge"
    emit.assert_awaited_once()


@pytest.mark.asyncio
async def test_merge_force_auto_merge_bypasses_global_gate():
    """force_auto_merge=True in decision.details merges even when
    settings.auto_merge_enabled is False. This is how the
    deterministic identifier rules (exact LEI/CIK/VAT/authority-id,
    SAME_AS cluster collapse, GLEIF successor) avoid filling the
    review queue with self-evident matches.
    """
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    decision = Decision(
        rule_name="exact_lei_match", action="merge", source_id="A", target_id="B",
        confidence=1.0, entity_type="Company",
        details={"force_auto_merge": True},
    )
    with patch.object(actions, "_merge", new=AsyncMock()) as merge_, \
         patch.object(actions, "_flag_same_as", new=AsyncMock()) as flag_, \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()), \
         patch.object(actions.settings, "auto_merge_enabled", False):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=decision,
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "auto_merge"
    merge_.assert_awaited_once()
    flag_.assert_not_called()


@pytest.mark.asyncio
async def test_merge_without_force_respects_global_gate():
    """No force_auto_merge stamp → respects the global gate.
    Confirms the bypass is opt-in, not a blanket override.
    """
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_merge", new=AsyncMock()) as merge_, \
         patch.object(actions, "_flag_same_as", new=AsyncMock()) as flag_, \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()), \
         patch.object(actions.settings, "auto_merge_enabled", False):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("merge"),
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "flag"
    flag_.assert_awaited_once()
    merge_.assert_not_called()


@pytest.mark.asyncio
async def test_link_does_not_emit_assert_same_as():
    """Link is not an equivalence — it's a typed edge (RELATED_TO etc.)."""
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    decision = Decision(
        rule_name="r", action="link", source_id="A", target_id="B",
        confidence=0.8, entity_type="Company", details={"rel_type": "RELATED_TO"},
    )
    with patch.object(actions, "_link", new=AsyncMock()), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=decision, entity=entity, candidate=candidate,
        )
    assert outcome == "auto_link"
    emit.assert_not_awaited()
