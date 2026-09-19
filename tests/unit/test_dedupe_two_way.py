"""merge_pair: two proposals for one pair become the one the consolidator
would have written had it seen both from the same side."""
from __future__ import annotations

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
