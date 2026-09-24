"""TranslationEnrichmentCohesionProject: the language Kohesio titles are in.

Kohesio publishes every project title in English whatever the country. The
first run in prod read the country instead, labelled English titles as
Lithuanian or Polish, and never asked for the country's own language.
"""
from __future__ import annotations

import pytest

from src.consolidator.clients.linguistics import EU_OFFICIAL_LANGS
from src.consolidator.rules.base import Entity
from src.consolidator.rules.cohesion.enrichment import infer_source_lang, missing_targets


def _project(**props) -> Entity:
    base = {"title": "Fund of Funds \"Innovation Promotion Fund\"", "detail_country": "LTU"}
    return Entity(entity_type="CohesionProject", id="kohesio-1", properties={**base, **props})


@pytest.mark.parametrize("country", ["LTU", "POL", "ROU", "FRA", None])
def test_a_kohesio_title_is_english_whatever_the_country(country):
    assert infer_source_lang(_project(detail_country=country)) == "en"


def test_the_countrys_own_language_is_requested():
    targets = missing_targets(_project())
    assert "lt" in targets and "en" not in targets
    assert len(targets) == len(EU_OFFICIAL_LANGS) - 1


def test_a_wrong_language_label_left_on_the_node_is_not_believed():
    assert infer_source_lang(_project(title_lang="lt")) == "en"
