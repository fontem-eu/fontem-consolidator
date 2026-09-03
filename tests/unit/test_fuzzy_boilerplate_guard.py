"""Fuzzy-name guards against shared legal-form boilerplate.

Jaro-*Winkler* pays a bonus for shared prefixes. When a jurisdiction's
long legal form is not stripped, it survives normalisation as a shared
prefix on every company in that country, and the score goes UP the
longer the boilerplate is. In prod this asserted one Latvian company
identical to 1,333 unrelated others at ~0.95 confidence — above the
0.92 flag threshold and, for some pairs, above the 0.97 auto-merge
threshold that DELETES a node.

Two defences, tested here:
  1. `_LEGAL_SUFFIX` strips the known long forms.
  2. `_shared_prefix_ok` is the structural backstop for the ones that
     are still missing from that wordlist — because a wordlist is
     never complete.
"""

import pytest
from rapidfuzz.distance import JaroWinkler

from src.config import settings
from src.consolidator.rules.company.fuzzy import (
    _normalise,
    _shared_prefix_ok,
)

LV = 'Sabiedrība ar ierobežotu atbildību "%s"'
THRESHOLD = settings.fuzzy_name_threshold


def _score(a: str, b: str) -> float:
    return JaroWinkler.normalized_similarity(_normalise(a), _normalise(b))


def _matches(a: str, b: str) -> bool:
    na, nb = _normalise(a), _normalise(b)
    if len(na) < settings.fuzzy_min_distinctive_chars:
        return False
    if len(nb) < settings.fuzzy_min_distinctive_chars:
        return False
    if JaroWinkler.normalized_similarity(na, nb) < THRESHOLD:
        return False
    return _shared_prefix_ok(na, nb, THRESHOLD)


@pytest.mark.parametrize(
    "name_a,name_b",
    [
        # Real pairs from prod, all previously asserted owl:sameAs.
        (LV % "AR TEXTIL", LV % "ZABBIX"),
        (LV % "AR TEXTIL", LV % "LIVONIA PRINT"),
        (LV % "ĶILUPE", LV % "Itelika"),
        (LV % "SupraMed", LV % "Tomega"),
    ],
)
def test_latvian_long_form_no_longer_collides(name_a, name_b):
    assert not _matches(name_a, name_b)


@pytest.mark.parametrize(
    "name_a,name_b",
    [
        (LV % "ZABBIX", 'SIA "ZABBIX"'),
        (LV % "AR TEXTIL", "AR TEXTIL SIA"),
    ],
)
def test_latvian_true_duplicates_still_match(name_a, name_b):
    """Stripping the long form is what lets the SHORT-form spelling of
    the same company match it. The fix must not cost recall."""
    assert _matches(name_a, name_b)


@pytest.mark.parametrize(
    "name_a,name_b",
    [
        ("Fayat", "FAYAT"),
        ("KNOWIT AS", "Knowit"),
        ("VEOLIA PROPRETE AQUITAINE", "VEOLIA PROPRETE AQUITAINE SAS"),
        ("ZIGA ZAGA, SL", "ZIGA ZAGA"),
        ("PI Vindija d.d.", "PI Vindija d.d. Varaždin"),
        ("G.K. PROFESSIONAL", "gk professional"),
    ],
)
def test_legitimate_variants_survive(name_a, name_b):
    """Sampled from prod edges that were correct. Guarding against
    boilerplate must not start rejecting real duplicates."""
    assert _matches(name_a, name_b)


def test_short_names_are_not_matched():
    """A 3-character name matches dozens of unrelated companies per
    country and the procurement-stub side carries no field that could
    ever resolve it, so it is noise a reviewer cannot action."""
    assert not _matches("mbs", "MBS")
    assert _matches("anios", "ANIOS")


@pytest.mark.parametrize(
    "distinct_a,distinct_b,expected",
    [
        ("AR TEXTIL", "ZABBIX", False),
        ("Itelika", "Lateca", False),
        ("Zabbix", "Zabbix doo", True),
    ],
)
def test_unknown_legal_form_is_caught_structurally(distinct_a, distinct_b, expected):
    """The backstop: a long legal form absent from `_LEGAL_SUFFIX`
    still inflates Jaro-Winkler above the threshold. The guard must
    reject on the part AFTER the shared prefix, while still allowing
    a genuine match that happens to share that prefix.
    """
    boiler = "Ograniceno Drustvo Sa Odgovornoscu Preduzece"
    a, b = f"{boiler} {distinct_a}", f"{boiler} {distinct_b}"
    # Precondition: without the guard these all pass the threshold.
    assert _score(a, b) >= THRESHOLD
    assert _matches(a, b) is expected


def test_guard_ignores_short_shared_prefixes():
    """Two names sharing only a few leading characters are ordinary
    near-matches, not boilerplate collisions."""
    assert _shared_prefix_ok("ACME LOGISTICS", "ACME LOGISTIC", THRESHOLD)
