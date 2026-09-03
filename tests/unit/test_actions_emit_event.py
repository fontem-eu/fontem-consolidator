"""Tests for the AssertSameAs emission contract.

``owl:sameAs`` is published for APPROVED equivalences only — an
auto-merge here, or a reviewer's approval in the queue. A ``flag`` or
``conflict`` outcome is a review *candidate* and must emit nothing.

These tests previously asserted the opposite. Emitting on flag put
1.34M unreviewed pairs into Virtuoso, of which the objectively
checkable subset (both sides carrying a national registration number)
was ~99% wrong. See actions.py's module docstring.
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
async def test_flag_does_not_emit_assert_same_as():
    """A flag is an unreviewed hypothesis. It queues a candidate and
    publishes nothing."""
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
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_conflict_does_not_emit_assert_same_as():
    """A conflict is the strongest signal we have that the two are NOT
    the same entity. Publishing owl:sameAs for it asserts the opposite
    of what detection concluded."""
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
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_downgraded_merge_does_not_emit():
    """auto_merge disabled turns a merge into a review candidate. It
    must not publish on the way down."""
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_flag_same_as", new=AsyncMock()), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit, \
         patch.object(actions.settings, "auto_merge_enabled", False):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("merge"),
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "flag"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_vetoed_merge_does_not_emit():
    """A :NOT_SAME_AS veto makes _merge a no-op. Emitting anyway would
    assert an equivalence for a merge that never happened."""
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_merge", new=AsyncMock(return_value=False)), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit, \
         patch.object(actions.settings, "auto_merge_enabled", True):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("merge"),
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "noop"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_vetoed_flag_reports_noop():
    """A vetoed flag wrote no edge, so the audit must not claim a
    candidate was queued."""
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_flag_same_as", new=AsyncMock(return_value=False)), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("flag"),
            entity=entity,
            candidate=candidate,
        )
    assert outcome == "noop"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_emits_assert_same_as_when_auto_merge_on():
    driver = MagicMock()
    candidate = MagicMock()
    entity = Entity(entity_type="Company", id="A", properties={})
    with patch.object(actions, "_merge", new=AsyncMock(return_value=True)), \
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
    with patch.object(actions, "_merge", new=AsyncMock(return_value=True)) as merge_, \
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
    with patch.object(actions, "_merge", new=AsyncMock(return_value=True)) as merge_, \
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
