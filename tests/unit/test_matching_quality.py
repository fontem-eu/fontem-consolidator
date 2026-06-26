"""Regression guard for the entity-resolution matcher's quality.

Pure-Python (no Neo4j): the name_country tier matches iff
apoc.text.clean(a) == clean(b), and eval.clean is a validated replica, so
the whole recall side runs in CI. Precision (false-merge) ground truth is
LEI homonyms — measured live by eval.live_report; here we pin the homonym
*awareness* so a refactor can't silently assume name+country is unique.
"""
from __future__ import annotations

import pytest

from src.consolidator.eval.clean import clean
from src.consolidator.eval.metrics import ConfusionMatrix
from src.consolidator.eval.perturb import perturbations

# Golden fixtures: real Company.name -> name_clean pulled from prod, chosen to
# exercise every clean() rule (German ü->ue, accent fold, ł kept, punctuation
# + whitespace + legal-form dots stripped). A few KB, not a corpus.
GOLDEN = [
    ("Müller KG", "muellerkg"),                              # German umlaut -> ue
    ("Mészáros és Mészáros Zrt.", "meszarosesmeszaroszrt"),  # Hungarian accents
    ("Société Générale S.A.", "societegeneralesa"),          # French accents
    ("O'Brien & Sons, Ltd.", "obriensonsltd"),               # apostrophe + &
    ("Łódź Sp. z o.o.", "łodzspzoo"),                        # ł kept, ó/ź fold
    ("Žďár", "zdar"),                                         # Czech carons
    ("NEURAXPHARM FRANCE ( Rang 1)", "neuraxpharmfrancerang1"),
    ("ACME  GmbH", "acmegmbh"),                              # double space
    ("VÝPRACHTICKÝ s.r.o.", "vyprachtickysro"),
    ("MAHLE Thermal and Fluid Systems Czechia s.r.o.",
     "mahlethermalandfluidsystemsczechiasro"),
]

# Real homonyms from prod: one (name_clean, country) -> many distinct LEIs.
# These ARE different companies; name_country would over-merge them, so the
# backfill must guard against it. We pin that they collapse to one clean.
HOMONYMS = ["Futura S.r.l.", "Futura srl", "FUTURA S.R.L."]


@pytest.mark.parametrize("name,expected", GOLDEN)
def test_clean_replica_matches_apoc(name, expected):
    assert clean(name) == expected


def test_clean_invariant_perturbations_preserve_match():
    """Accent/case/punct/whitespace noise must NOT break a name_country
    match — this is the regression guard on clean(). If clean() ever stops
    folding an accent or stripping a space, recall here drops below 1.0."""
    cm = _eval_recall([n for n, _ in GOLDEN], invariant_only=True)
    assert cm.recall == 1.0, cm.as_dict()
    assert cm.fn == 0


WORD_LEVEL = {"expand_legal_form", "abbrev_legal_form", "ampersand_to_and"}


def test_word_level_drift_is_a_known_blind_spot():
    """Legal-form expansion / '& -> and' clean to a DIFFERENT string, so
    name_country CANNOT link them (only the fuzzy tier / a shared identifier
    can). Pin this so the blind spot stays visible rather than silently
    'fixed' by a clean() change that would also wreck precision."""
    misses = 0
    for name, _ in GOLDEN:
        for p in perturbations(name):
            if p.kind in WORD_LEVEL:
                assert clean(p.name) != clean(name)
                misses += 1
    assert misses > 0  # at least one legal-form/ampersand case exercised


def test_german_umlaut_drop_is_a_recall_gap():
    """apoc.text.clean folds ü->ue, so the "ue" transliteration matches but a
    bare-"u" drop does not. A real recall gap the fuzzy tier must cover."""
    assert clean("Mueller") == clean("Müller")   # ue-form matches
    assert clean("Muller") != clean("Müller")    # bare-u form is a real miss


def test_homonyms_collapse_to_one_clean():
    cleans = {clean(n) for n in HOMONYMS}
    assert len(cleans) == 1, f"expected one clean, got {cleans}"


def test_confusion_matrix_metrics():
    cm = ConfusionMatrix(tp=98, fp=2, fn=10, tn=1000)
    assert round(cm.precision, 2) == 0.98
    assert round(cm.recall, 2) == 0.91
    assert ConfusionMatrix(0, 0, 0, 5).precision == 1.0  # no-decision = vacuous


def _eval_recall(names, *, invariant_only: bool) -> ConfusionMatrix:
    tp = fn = 0
    for name in names:
        base = clean(name)
        for p in perturbations(name):
            if invariant_only and not p.clean_invariant:
                continue
            if clean(p.name) == base:
                tp += 1
            else:
                fn += 1
    return ConfusionMatrix(tp=tp, fp=0, fn=fn, tn=0)


class _FakeSession:
    """Returns canned rows per query so live_report.report() composition is
    covered in CI without a Neo4j."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **_params):
        if "homonym_groups" in query:
            return [{"groups": 100, "homonym_groups": 2, "entities_at_risk": 7}]
        if "distinct_leis" in query:
            return [{"name_clean": "futurasrl", "country": "ITA", "distinct_leis": 51}]
        if "same_entity_name_drift_pairs" in query:
            return [{"same_entity_name_drift_pairs": 239}]
        # fidelity sample: one matching, one diverging
        return [{"name": "Müller KG", "name_clean": "muellerkg"},
                {"name": "X", "name_clean": "WRONG"}]


class _FakeDriver:
    def session(self):
        return _FakeSession()

    def close(self):
        pass


def test_live_report_composition():
    from src.consolidator.eval import live_report  # pylint: disable=import-outside-toplevel
    rep = live_report.report(_FakeDriver(), sample=2)
    assert rep["precision"]["group_precision"] == 0.98
    assert rep["precision"]["entities_at_risk"] == 7
    assert rep["worst_homonyms"][0]["distinct_leis"] == 51
    assert rep["recall_gap"]["same_entity_name_drift_pairs"] == 239
    assert rep["replica_fidelity"] == {"checked": 2, "match": 1, "pct": 50.0}
