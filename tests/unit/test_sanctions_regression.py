"""Forward-looking regression test for sanction-matching rules.

Today the consolidator does NOT carry any sanction-matching rule —
the eu_consolidated SANCTIONED edges are produced by the ETL
(`edgar-gmr-etl/src/etl/load_eu_sanctions.py`). This test pins that
fact so we notice if and when sanction matching moves into the
consolidator: at that point, the new rule MUST handle the historical
false-positive cases below.

Background: in production we observed 8 `SANCTIONED` edges, all of
them false positives where a 3-4 letter Company name like "AMD" /
"TSA" / "CRL" / "LRA" / "NADA" naively matched a sanction record
whose primary `name` was the same short code (the actual entity
name lived in `aliases`). All 8 companies were unrelated EU
entities — defamation risk if surfaced in the public UI. See PR
notes on `edgar-gmr-etl#fix/sanctions-name-match-guard` for the
full story.

If we ever add a sanction-matching rule to this repo:
- it MUST require `len(name) >= 6` (no acronym matches)
- it MUST require country/nationality agreement
- it MUST never mark a sanction match as auto-merge — flag-for-
  review only, with `reviewed: false` until a human signs off
- this file's parametrized cases must be regression-asserted to
  produce zero matches
"""
from __future__ import annotations

import pytest

# Five (six pairs) historical false positives — same set as the ETL
# regression suite. If you add a sanction-matching rule, parametrize a
# new test in this file with the cases imported from here.
SANCTION_REGRESSION_CASES = [
    # (company_name, company_country, sanction_short_name, sanction_nationality, why)
    pytest.param(
        "AMD", "FR", "AMD", "IR",
        id="AMD-FR-vs-Iranian-Aran-Modern-Devices",
    ),
    pytest.param(
        "TSA", "DK", "TSA", "IR",
        id="TSA-DK-vs-Iran-Centrifuge-Tech",
    ),
    pytest.param(
        "TSA", "FR", "TSA", "IR",
        id="TSA-FR-vs-Iran-Centrifuge-Tech",
    ),
    pytest.param(
        "CRL", "FR", "CRL", "IR",
        id="CRL-FR-vs-Iran-Composites",
    ),
    pytest.param(
        "LRA", "FR", "LRA", "UG",
        id="LRA-FR-vs-Lords-Resistance-Army",
    ),
    pytest.param(
        "NADA", "BE", "NADA", "KP",
        id="NADA-BE-vs-DPRK-Aerospace",
    ),
]


def test_no_consolidator_rule_targets_sanctioned_entity():
    """No registered rule may target SanctionedEntity. The moment one
    does, this test goes red and the author must add a sanction-
    specific guard test that exercises SANCTION_REGRESSION_CASES."""
    from src.consolidator.rules.loader import load_all  # pylint: disable=import-outside-toplevel
    from src.consolidator.rules.registry import list_rules  # pylint: disable=import-outside-toplevel

    load_all()
    offending = [
        r.name for r in list_rules()
        if "SanctionedEntity" in r.entity_types
    ]
    assert not offending, (
        f"Rules targeting SanctionedEntity: {offending}. If this is "
        "intentional, also add a parametrized test that runs each rule "
        "against SANCTION_REGRESSION_CASES and asserts zero auto-merge "
        "matches."
    )


def test_no_consolidator_rule_writes_sanctioned_action():
    """The Action literal allows {merge, link, flag, noop, enrich}. If
    someone widens it to include 'sanction' (or similar), this test
    ensures the regression-case suite is updated alongside."""
    from src.consolidator.rules.base import Action  # pylint: disable=import-outside-toplevel

    # `Action` is a typing.Literal — its args expose the allowed values
    allowed = set(getattr(Action, "__args__", ()))
    # The set is small and stable; pin it so widening it forces a
    # deliberate code-review touch.
    assert allowed == {"merge", "link", "flag", "noop", "enrich"}, (
        f"Action literal changed to {allowed}. If a sanction-specific "
        "action is being added, also extend test_sanctions_regression.py "
        "with a parametrized case-by-case guard against SANCTION_REGRESSION_CASES."
    )


@pytest.mark.parametrize(
    "company_name,company_country,sanction_name,sanction_nationality",
    SANCTION_REGRESSION_CASES,
)
def test_short_name_acronym_invariant(
    company_name, company_country, sanction_name, sanction_nationality,
):
    """All historical false-positive matches were on names of length
    ≤ 4 — too short to be specific. This invariant captures the rule
    we'd want any future sanction-matching rule to enforce."""
    assert len(company_name) < 6, (
        f"{company_name!r} is the regression-case company name; if it's "
        "longer than 6 chars, the case no longer reproduces the "
        "acronym-collision class of bug."
    )
    assert len(sanction_name) < 6
    assert company_country != sanction_nationality, (
        f"For {company_name!r} the regression case requires that "
        "company country and sanction nationality differ — that's the "
        "whole point. If they happen to match, this is no longer a "
        "false-positive case."
    )
