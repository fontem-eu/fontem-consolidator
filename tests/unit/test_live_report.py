"""Tests for eval.live_report — the on-demand matching-quality report.

Untested until now, and it is arithmetic that gets quoted in decisions:
group_precision and replica-fidelity percentages are what tell us whether
a blind name_country backfill would over-merge. A silent divide-by-zero or
an inverted ratio here produces a confident, wrong number.

The driver is stubbed — no live graph required.
"""
from __future__ import annotations

# pylint: disable=missing-function-docstring,protected-access

from unittest.mock import MagicMock, patch

from src.consolidator.eval import live_report


def _driver(results):
    """results: list of row-lists, returned in query order."""
    session = MagicMock()
    session.run = MagicMock(side_effect=[iter(r) for r in results])
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=session)
    ctx.__exit__ = MagicMock(return_value=None)
    driver = MagicMock()
    driver.session = MagicMock(return_value=ctx)
    return driver


def _rows(*dicts):
    return [dict(d) for d in dicts]


def test_run_materialises_records_as_dicts():
    session = MagicMock()
    session.run = MagicMock(return_value=iter([{"a": 1}, {"a": 2}]))
    assert live_report._run(session, "Q") == [{"a": 1}, {"a": 2}]


def test_report_computes_group_precision():
    """precision = clean groups / all groups. 8 of 10 clean -> 0.8."""
    driver = _driver([
        _rows({"groups": 10, "homonym_groups": 2, "entities_at_risk": 5}),
        _rows({"name_clean": "acme", "country": "PT", "distinct_leis": 3}),
        _rows({"same_entity_name_drift_pairs": 7}),
        _rows({"name": "ACME S.A.", "name_clean": "acme sa"}),
    ])
    with patch.object(live_report, "clean", return_value="acme sa"):
        out = live_report.report(driver)
    assert out["precision"]["group_precision"] == 0.8
    assert out["precision"]["entities_at_risk"] == 5
    assert out["worst_homonyms"][0]["distinct_leis"] == 3
    assert out["recall_gap"]["same_entity_name_drift_pairs"] == 7


def test_report_treats_an_empty_graph_as_perfect_precision():
    """No groups means nothing to get wrong; dividing would raise instead."""
    driver = _driver([
        _rows({"groups": 0, "homonym_groups": 0, "entities_at_risk": 0}),
        [],
        _rows({"same_entity_name_drift_pairs": 0}),
        [],
    ])
    out = live_report.report(driver)
    assert out["precision"]["group_precision"] == 1.0


def test_report_scores_replica_fidelity_against_the_materialised_column():
    """This is the apoc-drift check: clean() must reproduce name_clean."""
    driver = _driver([
        _rows({"groups": 4, "homonym_groups": 0, "entities_at_risk": 0}),
        [],
        _rows({"same_entity_name_drift_pairs": 0}),
        _rows({"name": "A", "name_clean": "a"},
              {"name": "B", "name_clean": "WRONG"}),
    ])
    with patch.object(live_report, "clean", side_effect=lambda n: n.lower()):
        out = live_report.report(driver)
    f = out["replica_fidelity"]
    assert (f["checked"], f["match"], f["pct"]) == (2, 1, 50.0)


def test_report_reports_no_fidelity_percentage_without_a_sample():
    """0/0 is not 0% — it is unknown, and must not read as total failure."""
    driver = _driver([
        _rows({"groups": 1, "homonym_groups": 0, "entities_at_risk": 0}),
        [], _rows({"same_entity_name_drift_pairs": 0}), [],
    ])
    out = live_report.report(driver)
    assert out["replica_fidelity"] == {"checked": 0, "match": 0, "pct": None}


def test_report_passes_the_sample_size_to_the_fidelity_query():
    driver = _driver([
        _rows({"groups": 1, "homonym_groups": 0, "entities_at_risk": 0}),
        [], _rows({"same_entity_name_drift_pairs": 0}), [],
    ])
    live_report.report(driver, sample=17)
    session = driver.session.return_value.__enter__.return_value
    assert session.run.call_args.kwargs == {"n": 17}


def test_main_closes_the_driver_even_when_the_report_raises():
    """The report runs against shared; a leaked session holds a connection."""
    driver = MagicMock()
    with patch.object(live_report.GraphDatabase, "driver", return_value=driver), \
         patch.dict("os.environ", {"NEO4J_URI": "bolt://x",
                                   "NEO4J_PASSWORD": "p"}, clear=False), \
         patch.object(live_report, "report", side_effect=RuntimeError("boom")):
        try:
            live_report.main()
        except RuntimeError:
            pass
    driver.close.assert_called_once()


def test_main_prints_the_headline_numbers(capsys):
    driver = MagicMock()
    rep = {
        "precision": {"group_precision": 0.9876, "homonym_groups": 3,
                      "entities_at_risk": 9},
        "worst_homonyms": [{"name_clean": "acme", "country": "PT",
                            "distinct_leis": 4}],
        "recall_gap": {"same_entity_name_drift_pairs": 11},
        "replica_fidelity": {"checked": 100, "match": 99, "pct": 99.0},
    }
    with patch.object(live_report.GraphDatabase, "driver", return_value=driver), \
         patch.dict("os.environ", {"NEO4J_URI": "bolt://x",
                                   "NEO4J_PASSWORD": "p"}, clear=False), \
         patch.object(live_report, "report", return_value=rep):
        live_report.main()
    out = capsys.readouterr().out
    assert "0.9876" in out
    assert "3 homonym groups / 9 entities" in out
    assert "'acme' (PT): 4 real companies" in out
    assert "11 pairs" in out
    assert "99/100 = 99.0%" in out
