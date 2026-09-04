"""Tests for the AssertSameAs emission contract.

``owl:sameAs`` is published for APPROVED equivalences only — an
auto-merge here, or a reviewer's approval in the queue. Everything else
writes a :SAME_AS_CANDIDATE, which is a proposal and emits nothing.

These tests previously asserted the opposite. Emitting on every match
put 1.34M unreviewed pairs into Virtuoso, of which the objectively
checkable subset (both sides carrying a national registration number)
was ~99% wrong. See actions.py's module docstring.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.consolidator import actions
from src.consolidator.rules.base import Decision


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
async def test_proposal_does_not_emit_assert_same_as():
    """A proposal is an unreviewed hypothesis. It writes a candidate
    edge and publishes nothing."""
    driver = MagicMock()
    with patch.object(actions, "_propose_candidate", new=AsyncMock()) as fsa, \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("flag"),
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
    with patch.object(actions, "_propose_candidate", new=AsyncMock()), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("flag", conflict=True),
        )
    assert outcome == "conflict"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_downgraded_merge_does_not_emit():
    """auto_merge disabled turns a merge decision into a review
    candidate. It must not publish on the way down."""
    driver = MagicMock()
    with patch.object(actions, "_propose_candidate", new=AsyncMock()), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit, \
         patch.object(actions.settings, "auto_merge_enabled", False):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("merge"),
        )
    assert outcome == "flag"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_settled_pair_reports_noop():
    """Already corrected, asserted or declined — no candidate edge was
    written, so the audit must not claim one was queued."""
    driver = MagicMock()
    with patch.object(actions, "_propose_candidate", new=AsyncMock(return_value=False)), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("flag"),
        )
    assert outcome == "noop"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_allowed_rule_emits_assert_same_as():
    driver = MagicMock()
    with          patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit, \
         patch.object(actions, "_is_settled", new=AsyncMock(return_value=False)), \
         patch.object(actions.settings, "auto_merge_enabled", True):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("merge"),
        )
    assert outcome == "auto_assert"
    emit.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_auto_merge_bypasses_global_gate():
    """force_auto_merge=True in decision.details merges even when
    settings.auto_merge_enabled is False. This is how the
    deterministic identifier rules (exact LEI/CIK/VAT/authority-id,
    SAME_AS cluster collapse, GLEIF successor) avoid filling the
    review queue with self-evident matches.
    """
    driver = MagicMock()
    decision = Decision(
        rule_name="exact_lei_match", action="merge", source_id="A", target_id="B",
        confidence=1.0, entity_type="Company",
        details={"force_auto_merge": True},
    )
    with          patch.object(actions, "_propose_candidate", new=AsyncMock()) as flag_, \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()), \
         patch.object(actions, "_is_settled", new=AsyncMock(return_value=False)), \
         patch.object(actions.settings, "auto_merge_enabled", False):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=decision,
        )
    assert outcome == "auto_assert"
    flag_.assert_not_called()


@pytest.mark.asyncio
async def test_without_force_respects_global_gate():
    """No force_auto_merge stamp → respects the global gate.
    Confirms the bypass is opt-in, not a blanket override.
    """
    driver = MagicMock()
    with          patch.object(actions, "_propose_candidate", new=AsyncMock()) as flag_, \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()), \
         patch.object(actions, "_is_settled", new=AsyncMock(return_value=False)), \
         patch.object(actions.settings, "auto_merge_enabled", False):
        outcome = await actions.execute(
            driver, "neo4j",
            decision=_decision("merge"),
        )
    assert outcome == "flag"
    flag_.assert_awaited_once()


@pytest.mark.asyncio
async def test_link_does_not_emit_assert_same_as():
    """Link is not an equivalence — it's a typed edge (RELATED_TO etc.)."""
    driver = MagicMock()
    decision = Decision(
        rule_name="r", action="link", source_id="A", target_id="B",
        confidence=0.8, entity_type="Company", details={"rel_type": "RELATED_TO"},
    )
    with patch.object(actions, "_link", new=AsyncMock()), \
         patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit:
        outcome = await actions.execute(
            driver, "neo4j",
            decision=decision,
        )
    assert outcome == "auto_link"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_settled_pair_asserts_nothing_again():
    """The re-emit loop this closes.

    Nothing is deleted any more, so exact_lei_match finds the same pair
    on every sweep forever. Merging used to hide that by removing the
    duplicate node; without a local settled record the run would emit a
    duplicate AssertSameAs for every deterministic match, every rotation,
    for the life of the graph.
    """
    driver = MagicMock()
    with patch.object(actions.eventlog, "emit_assert_same_as", new=AsyncMock()) as emit, \
         patch.object(actions, "_is_settled", new=AsyncMock(return_value=True)), \
         patch.object(actions.settings, "auto_merge_enabled", True):
        outcome = await actions.execute(
            driver, "neo4j", decision=_decision("merge"),
        )
    assert outcome == "noop"
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_settled_check_is_skipped_when_not_allowed_to_assert():
    """A rule that cannot assert goes straight to proposing; the
    candidate write has its own guards and the extra read would be
    wasted on every flag in the sweep."""
    driver = MagicMock()
    with patch.object(actions, "_propose_candidate",
                      new=AsyncMock(return_value=True)), \
         patch.object(actions, "_is_settled", new=AsyncMock()) as settled, \
         patch.object(actions.settings, "auto_merge_enabled", False):
        outcome = await actions.execute(
            driver, "neo4j", decision=_decision("merge"),
        )
    assert outcome == "flag"
    settled.assert_not_awaited()
