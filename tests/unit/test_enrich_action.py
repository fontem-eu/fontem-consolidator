"""Tests for the _enrich action executor — specifically the encoder-id
invariants and the prop-name layout on the Cypher write.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.consolidator import actions
from src.consolidator.rules.base import Decision


pytestmark = pytest.mark.asyncio


def _decision(**overrides) -> Decision:
    details = {
        "field": "name",
        "translations": {},
        "embedding": None,
        "embedding_encoder": None,
        "source_lang": None,
    }
    details.update(overrides.pop("details", {}))
    base = dict(
        rule_name="translation_enrichment_authority",
        action="enrich",
        source_id="AUTH-1",
        target_id="AUTH-1",
        confidence=1.0,
        entity_type="Authority",
        details=details,
    )
    base.update(overrides)
    return Decision(**base)


def _capturing_driver():
    """AsyncDriver stub that captures the (cypher, params) of .session().run()."""
    captured: dict = {}

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    async def _run(cypher, **params):
        captured["cypher"] = cypher
        captured["params"] = params
    session.run = _run

    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver, captured


async def test_enrich_raises_when_embedding_has_no_encoder_id():
    driver, _ = _capturing_driver()
    d = _decision(details={
        "embedding": [0.1, 0.2, 0.3],
        "embedding_encoder": None,  # missing → must raise
    })
    with pytest.raises(ValueError, match="embedding_encoder"):
        await actions._enrich(driver, "neo4j", decision=d)


async def test_enrich_writes_encoder_and_dim_alongside_embedding():
    driver, captured = _capturing_driver()
    d = _decision(details={
        "embedding": [0.1] * 768,
        "embedding_encoder": "labse@1.0.0-836121a",
    })
    await actions._enrich(driver, "neo4j", decision=d)

    props = captured["params"]["props"]
    assert props["name_embedding"] == [0.1] * 768
    assert props["name_embedding_encoder"] == "labse@1.0.0-836121a"
    assert props["name_embedding_dim"] == 768


async def test_enrich_translations_only_does_not_require_encoder_id():
    """Translations without an embedding don't need an encoder id."""
    driver, captured = _capturing_driver()
    d = _decision(details={
        "translations": {"en": "Hello", "fr": "Bonjour"},
        "source_lang": "de",
    })
    await actions._enrich(driver, "neo4j", decision=d)

    props = captured["params"]["props"]
    assert props["name_en"] == "Hello"
    assert props["name_fr"] == "Bonjour"
    assert props["name_lang"] == "de"
    assert "name_embedding" not in props
    assert "name_embedding_encoder" not in props


async def test_enrich_contract_field_writes_title_prefix():
    """Contract rule sets field="title" — props use title_* keys."""
    driver, captured = _capturing_driver()
    d = _decision(entity_type="Contract", details={
        "field": "title",
        "translations": {"en": "Winter service 2025ff"},
        "source_lang": "de",
    })
    await actions._enrich(driver, "neo4j", decision=d)

    props = captured["params"]["props"]
    assert props["title_en"] == "Winter service 2025ff"
    assert props["title_lang"] == "de"
    assert "name_en" not in props
