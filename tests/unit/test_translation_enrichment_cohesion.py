"""TranslationEnrichmentCohesionProject: the language Kohesio titles are in.

The loader states title_lang="en" when a title came from Kohesio's English-name
column, and nothing when it came from the programme-language fallback. The rule
takes that statement or asks the model; it never infers from the country (the
first prod run labelled English titles Lithuanian that way) and never assumes
English, which would mislabel every fallback title.
"""
from __future__ import annotations

import pytest

from src.consolidator.clients.linguistics import EU_OFFICIAL_LANGS
from src.consolidator.rules.base import Entity
from src.consolidator.rules.cohesion.enrichment import (
    UNDETERMINED, infer_source_lang, missing_targets,
)


def _project(**props) -> Entity:
    base = {"title": "Fund of Funds \"Innovation Promotion Fund\"", "detail_country": "LTU"}
    return Entity(entity_type="CohesionProject", id="kohesio-1", properties={**base, **props})


@pytest.mark.parametrize("country", ["LTU", "POL", "ROU", "FRA", None])
def test_a_stated_english_title_is_english_whatever_the_country(country):
    assert infer_source_lang(_project(detail_country=country, title_lang="en")) == "en"


@pytest.mark.parametrize("country", ["LTU", "POL", None])
def test_an_unstated_title_is_left_to_the_model(country):
    assert infer_source_lang(_project(detail_country=country)) == UNDETERMINED


def test_the_countrys_own_language_is_requested():
    targets = missing_targets(_project(title_lang="en"))
    assert "lt" in targets and "en" not in targets
    assert len(targets) == len(EU_OFFICIAL_LANGS) - 1


def test_an_unknown_source_asks_for_every_language():
    assert missing_targets(_project()) == list(EU_OFFICIAL_LANGS)
