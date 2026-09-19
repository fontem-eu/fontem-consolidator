"""merge_pair: two proposals for one pair become the one the consolidator
would have written had it seen both from the same side."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.consolidator import dedupe_two_way as dd
from src.consolidator.dedupe_two_way import merge_pair


def _pending(rules, confs, dates, **extra):
    top = max(range(len(confs)), key=lambda i: (confs[i], -i))
    return {"status": "pending", "detection_rules": rules, "detection_confidences": confs,
            "detection_dates": dates, "method": rules[top], "confidence": confs[top],
            "detected_at": dates[top], "conflict": False, **extra}


def test_the_approved_edge_survives_and_keeps_its_assertion():
    approved = {"status": "approved", "origin": "auto", "method": "fuzzy_name_same_country",
                "confidence": 1.0, "decided_at": "2026-09-18T18:07:49",
                "detection_rules": ["exact_name_country_match"], "detection_confidences": [0.95],
                "detection_dates": ["2026-09-18T18:07:48"], "detected_at": "2026-09-18T18:07:48",
                "conflict": False}
    pending = _pending(["exact_name_country_match", "fuzzy_name_same_country"], [0.95, 0.97],
                       ["2026-09-18T17:11:04", "2026-09-18T19:00:00"])
    keep, props = merge_pair(pending, approved)
    assert keep == 1
    assert props["status"] == "approved"
    # the assertion's own fields are not recomputed from the proposals
    assert (props["method"], props["confidence"], props["origin"]) == (
        "fuzzy_name_same_country", 1.0, "auto")
    assert props["decided_at"] == "2026-09-18T18:07:49"
    # the history is the union, one entry per rule, the latest firing
    assert props["detection_rules"] == ["exact_name_country_match", "fuzzy_name_same_country"]
    assert props["detection_dates"] == ["2026-09-18T18:07:48", "2026-09-18T19:00:00"]


def test_between_two_pending_the_older_survives_and_the_summary_is_recomputed():
    older = _pending(["exact_name_country_match"], [0.95], ["2026-09-18T18:39:56"])
    newer = _pending(["fuzzy_name_same_country"], [0.99], ["2026-09-18T21:18:53"])
    keep, props = merge_pair(newer, older)
    assert keep == 1
    assert props["status"] == "pending"
    assert props["detection_rules"] == ["exact_name_country_match", "fuzzy_name_same_country"]
    assert (props["method"], props["confidence"], props["detected_at"]) == (
        "fuzzy_name_same_country", 0.99, "2026-09-18T21:18:53")


def test_a_rule_on_both_keeps_its_latest_firing():
    a = _pending(["exact_name_country_match"], [0.90], ["2026-09-18T10:00:00"])
    b = _pending(["exact_name_country_match"], [0.95], ["2026-09-18T12:00:00"])
    _, props = merge_pair(a, b)
    assert props["detection_rules"] == ["exact_name_country_match"]
    assert props["detection_confidences"] == [0.95]
    assert props["detection_dates"] == ["2026-09-18T12:00:00"]


def test_a_confidence_tie_goes_to_the_earlier_rule():
    a = _pending(["rule_a"], [0.95], ["2026-09-18T10:00:00"])
    b = _pending(["rule_b"], [0.95], ["2026-09-18T11:00:00"])
    _, props = merge_pair(a, b)
    assert props["method"] == "rule_a"


def test_between_two_approved_the_first_assertion_survives():
    first = {"status": "approved", "decided_at": "2026-09-18T10:00:00", "method": "m1",
             "confidence": 0.99, "detection_rules": ["m1"], "detection_confidences": [0.99],
             "detection_dates": ["2026-09-18T09:59:00"]}
    second = {"status": "approved", "decided_at": "2026-09-18T11:00:00", "method": "m2",
              "confidence": 0.98, "detection_rules": ["m2"], "detection_confidences": [0.98],
              "detection_dates": ["2026-09-18T10:59:00"]}
    keep, props = merge_pair(second, first)
    assert keep == 1
    assert props["method"] == "m1" and props["decided_at"] == "2026-09-18T10:00:00"


def test_a_conflict_on_either_side_survives_with_its_explanation():
    plain = _pending(["rule_a"], [0.9], ["2026-09-18T10:00:00"])
    contested = _pending(["rule_b"], [0.9], ["2026-09-18T11:00:00"], conflict=True,
                         conflict_property="lei", conflict_left="LEI-1", conflict_right="LEI-2")
    keep, props = merge_pair(plain, contested)
    assert keep == 0
    assert props["conflict"] is True
    assert (props["conflict_property"], props["conflict_left"], props["conflict_right"]) == (
        "lei", "LEI-1", "LEI-2")


def test_merging_is_symmetric():
    a = _pending(["rule_a"], [0.9], ["2026-09-18T10:00:00"])
    b = _pending(["rule_b"], [0.95], ["2026-09-18T11:00:00"])
    ka, pa = merge_pair(a, b)
    kb, pb = merge_pair(b, a)
    assert pa == pb and ka != kb


# --------------------------------------------------------------------------
# The command around merge_pair: reads, batches, the optimistic write.
# --------------------------------------------------------------------------
class _Result:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        self._it = iter(self._rows)  # pylint: disable=attribute-defined-outside-init
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def single(self):
        return self._rows[0] if self._rows else None


class _Graph:
    """Two-way pairs held in memory; answers the command's four queries."""

    def __init__(self, pairs, *, fold_nothing=False, same_as=0):
        self.pairs = list(pairs)
        self.fold_nothing = fold_nothing
        self.same_as = same_as
        self.writes = 0

    def run(self, query, **params):
        if "RETURN elementId(r1) AS id1" in query:
            rows = self.pairs[: params["batch"]] if params.get("batch") else list(self.pairs)
            return _Result(rows)
        if "UNWIND $rows" in query:
            self.writes += 1
            if self.fold_nothing:
                return _Result([{"folded": 0}])
            done = {r["keep"] for r in params["rows"]} | {r["drop"] for r in params["rows"]}
            before = len(self.pairs)
            self.pairs = [p for p in self.pairs if p["id1"] not in done]
            return _Result([{"folded": before - len(self.pairs)}])
        if ":SAME_AS]" in query:
            dry = "RETURN count(*) AS pairs" in query
            n, self.same_as = self.same_as, (self.same_as if dry else 0)
            return _Result([[n]])
        raise AssertionError(query)


class _Driver:
    def __init__(self, graph):
        self.graph = graph

    def session(self, database=None):  # pylint: disable=unused-argument
        graph = self.graph

        class _S:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return False

            async def run(self, query, **params):
                return graph.run(query, **params)
        return _S()


def _edge(rule, date, status):
    return {"status": status, "detection_rules": [rule], "detection_confidences": [0.95],
            "detection_dates": [date], "method": rule, "confidence": 0.95, "detected_at": date,
            "conflict": False}


def _pair(i, s1="pending", s2="pending"):
    return {"id1": f"r{i}a", "p1": _edge("rule_a", "2026-09-18T10:00:00", s1),
            "id2": f"r{i}b", "p2": _edge("rule_b", "2026-09-18T11:00:00", s2)}


def test_a_dry_run_reads_everything_and_writes_nothing():
    graph = _Graph([_pair(i) for i in range(5)] + [_pair(9, "approved", "approved")])
    stats = asyncio.run(dd.fold_candidates(_Driver(graph), "neo4j", batch=2, dry_run=True))
    assert stats["pairs"] == 6 and stats["folded"] == 0
    assert stats["both_approved"] == 1 and stats["kept_approved"] == 1
    assert graph.writes == 0 and len(graph.pairs) == 6


def test_a_run_folds_batch_by_batch_until_none_are_left():
    graph = _Graph([_pair(i) for i in range(5)])
    stats = asyncio.run(dd.fold_candidates(_Driver(graph), "neo4j", batch=2, dry_run=False))
    assert stats["folded"] == 5 and not graph.pairs
    assert graph.writes == 3  # 2 + 2 + 1


def test_a_pass_that_folds_nothing_stops_rather_than_spinning():
    """Everything left changed underneath (the optimistic check refused it):
    leave it for a re-run."""
    graph = _Graph([_pair(i) for i in range(3)], fold_nothing=True)
    stats = asyncio.run(dd.fold_candidates(_Driver(graph), "neo4j", batch=10, dry_run=False))
    assert stats["folded"] == 0 and graph.writes == 1


def test_write_rows_carry_what_was_read_for_the_optimistic_check():
    rows, _ = dd._plan([_pair(1, "pending", "approved")])  # pylint: disable=protected-access
    row = rows[0]
    assert (row["keep"], row["drop"]) == ("r1b", "r1a")  # the approved one stays
    assert row["keep_status"] == "approved" and row["drop_status"] == "pending"
    assert row["keep_dates"] == ["2026-09-18T11:00:00"]
    assert row["drop_dates"] == ["2026-09-18T10:00:00"]


def test_same_as_pairs_are_counted_on_a_dry_run_and_folded_on_a_run():
    graph = _Graph([], same_as=15)
    assert asyncio.run(dd.fold_same_as(_Driver(graph), "neo4j", dry_run=True)) == 15
    assert graph.same_as == 15
    assert asyncio.run(dd.fold_same_as(_Driver(graph), "neo4j", dry_run=False)) == 15
    assert graph.same_as == 0


def test_run_reports_and_always_closes_the_driver():
    graph = _Graph([_pair(1)], same_as=2)
    close = AsyncMock()
    with patch.object(dd, "get_driver", AsyncMock(return_value=_Driver(graph))), \
         patch.object(dd, "close_driver", close):
        asyncio.run(dd.run(batch=10, dry_run=False))
    close.assert_awaited_once()
    assert not graph.pairs and graph.same_as == 0


@pytest.mark.parametrize("argv,dry,batch", [
    ([], False, 1000),
    (["--dry-run", "--batch", "50"], True, 50),
])
def test_cli(argv, dry, batch):
    with patch.object(dd, "run", AsyncMock()) as run:
        dd.main(argv)
    run.assert_awaited_once_with(batch=batch, dry_run=dry)
