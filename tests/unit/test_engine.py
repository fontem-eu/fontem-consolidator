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


class _FakeRuleWithThreshold(_FakeRule):
    """Rule that sets auto_merge_threshold so the engine can promote
    a high-confidence flag → merge. Per-test confidence and conflict
    flag are passed via constructor args."""
    name = "fake_with_threshold"
    auto_merge_threshold = 0.95

    def __init__(self, conf: float = 0.9, conflict: bool = False) -> None:
        self.conf = conf
        self.conflict = conflict

    async def resolve(self, entity, candidate):
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.conf,
            entity_type="Company",
            details={"conflict": self.conflict},
        )

    async def find_candidates(self, entity):
        return [
            Candidate(
                entity=Entity("Company", "gmr-B", {"name": "B"}),
                context={},
            )
        ]


@pytest.mark.asyncio
async def test_engine_promotes_flag_to_merge_above_threshold():
    """Confidence above the rule's auto_merge_threshold AND no
    conflict → engine rewrites the Decision from flag to merge before
    dispatching to actions.execute."""
    rule = _FakeRuleWithThreshold(conf=0.97, conflict=False)
    captured_action = []

    async def _capture(_driver, _database, *, decision, entity, candidate, **_):
        del entity, candidate  # unused
        captured_action.append(decision.action)
        return "auto_merge"

    with patch("src.consolidator.engine.list_rules", return_value=[rule]), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {"name": "A"})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-x")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ), patch(
        "src.consolidator.engine.actions.execute", _capture
    ):
        result = await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="gmr-A"
        )

    # Engine rewrote action from "flag" to "merge"
    assert captured_action == ["merge"]
    assert result.decisions[0]["action"] == "merge"


@pytest.mark.asyncio
async def test_engine_does_not_promote_below_threshold():
    rule = _FakeRuleWithThreshold(conf=0.94, conflict=False)  # below 0.95
    captured_action = []

    async def _capture(_driver, _database, *, decision, entity, candidate, **_):
        del entity, candidate
        captured_action.append(decision.action)
        return "flag"

    with patch("src.consolidator.engine.list_rules", return_value=[rule]), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {"name": "A"})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-y")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ), patch(
        "src.consolidator.engine.actions.execute", _capture
    ):
        await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="gmr-A"
        )

    # Stays as flag — confidence didn't clear the threshold
    assert captured_action == ["flag"]


@pytest.mark.asyncio
async def test_engine_does_not_promote_when_conflict_set():
    """Even at maximum confidence, a hard ID conflict (mismatched
    LEI / VAT / etc.) keeps the pair in the human-review queue."""
    rule = _FakeRuleWithThreshold(conf=1.0, conflict=True)  # conflict overrides
    captured_action = []

    async def _capture(_driver, _database, *, decision, entity, candidate, **_):
        del entity, candidate
        captured_action.append(decision.action)
        return "flag"

    with patch("src.consolidator.engine.list_rules", return_value=[rule]), patch(
        "src.consolidator.engine.entities.load",
        AsyncMock(return_value=Entity("Company", "gmr-A", {"name": "A"})),
    ), patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value="run-z")), patch(
        "src.consolidator.engine.audit.end_run", AsyncMock()
    ), patch(
        "src.consolidator.engine.audit.record_decision", AsyncMock()
    ), patch(
        "src.consolidator.engine.actions.execute", _capture
    ):
        await engine.consolidate(
            AsyncMock(), "neo4j", entity_type="Company", entity_id="gmr-A"
        )

    assert captured_action == ["flag"]


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


class _EnrichRule(_FakeRule):
    """Enrich-action rule used to verify the match_only / enrich_only mode
    filter. Targets the source entity itself (legitimate for enrichment)."""

    name = "fake_enrich"
    action = "enrich"

    async def find_candidates(self, entity):
        return [Candidate(entity=entity, context={})]

    async def resolve(self, entity, candidate):
        return Decision(
            rule_name=self.name,
            action="enrich",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=1.0,
            entity_type="Company",
            details={},
        )


def _patch_engine_deps(rules, run_id, capture_fn):
    """Common scaffolding for the three mode tests below."""
    return [
        patch("src.consolidator.engine.list_rules", return_value=rules),
        patch(
            "src.consolidator.engine.entities.load",
            AsyncMock(return_value=Entity("Company", "gmr-A", {"name": "A"})),
        ),
        patch("src.consolidator.engine.audit.start_run", AsyncMock(return_value=run_id)),
        patch("src.consolidator.engine.audit.end_run", AsyncMock()),
        patch("src.consolidator.engine.audit.record_decision", AsyncMock()),
        patch("src.consolidator.engine.actions.execute", capture_fn),
    ]


async def _run_with_mode(rules, run_id, mode):
    fired: list[str] = []

    async def _capture(_d, _db, *, decision, entity, candidate, **_):
        del entity, candidate
        fired.append(decision.rule_name)
        return decision.action

    patches = _patch_engine_deps(rules, run_id, _capture)
    for p in patches:
        p.start()
    try:
        kwargs = {"entity_type": "Company", "entity_id": "gmr-A"}
        if mode is not None:
            kwargs["mode"] = mode
        await engine.consolidate(AsyncMock(), "neo4j", **kwargs)
    finally:
        for p in patches:
            p.stop()
    return fired


@pytest.mark.asyncio
async def test_engine_match_only_skips_enrich_rules():
    """mode='match_only' runs dedup/match rules but skips enrich rules.
    Used by the dedup sweep so it isn't blocked on linguistics RTTs."""
    fired = await _run_with_mode([_FakeRule(), _EnrichRule()], "run-mo", "match_only")
    assert fired == ["fake"]


@pytest.mark.asyncio
async def test_engine_enrich_only_skips_match_rules():
    """mode='enrich_only' runs translation/enrichment rules only.
    Used by the translation-backfill sweep."""
    fired = await _run_with_mode([_FakeRule(), _EnrichRule()], "run-eo", "enrich_only")
    assert fired == ["fake_enrich"]


@pytest.mark.asyncio
async def test_engine_default_mode_runs_everything():
    """mode='all' is the default and preserves the existing behaviour of
    running every applicable rule."""
    fired = await _run_with_mode([_FakeRule(), _EnrichRule()], "run-all", None)
    assert sorted(fired) == ["fake", "fake_enrich"]


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
