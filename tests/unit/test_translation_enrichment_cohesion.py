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
def test_an_unstated_title_waits_for_the_backfill_to_detect_it(country):
    assert infer_source_lang(_project(detail_country=country)) is None
    assert missing_targets(_project(detail_country=country)) == []


def test_a_kept_detection_is_used():
    assert infer_source_lang(_project(title_lang_detected="pl")) == "pl"


def test_the_countrys_own_language_is_requested():
    targets = missing_targets(_project(title_lang="en"))
    assert "lt" in targets and "en" not in targets
    assert len(targets) == len(EU_OFFICIAL_LANGS) - 1


def test_a_language_outside_the_24_asks_for_every_language():
    assert infer_source_lang(_project(title_lang="no")) == UNDETERMINED
    assert missing_targets(_project(title_lang="no")) == list(EU_OFFICIAL_LANGS)


@pytest.mark.asyncio
async def test_live_translation_stops_below_the_grant_cut_off():
    from src.consolidator.rules.cohesion.enrichment import (  # pylint: disable=import-outside-toplevel
        TranslationEnrichmentCohesionProject,
    )
    rule = TranslationEnrichmentCohesionProject()
    assert await rule.applies(_project(title_lang="en", detail_eu_contribution=70_000_000.0))
    assert not await rule.applies(_project(title_lang="en", detail_eu_contribution=69_999_999.0))
    assert not await rule.applies(_project(title_lang="en"))
