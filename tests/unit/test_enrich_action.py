"""Tests for the _enrich action executor — specifically the encoder-id
invariants and the prop-name layout on the Cypher write.
"""
# protected-access: tests drive the action by calling actions._enrich
# directly — pinning the encoder/embedding cypher contract that the
# public execute() dispatcher routes to.
# pylint: disable=protected-access
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
    base = {
        "rule_name": "authority_name_embedding",
        "action": "enrich",
        "source_id": "AUTH-1",
        "target_id": "AUTH-1",
        "confidence": 1.0,
        "entity_type": "Authority",
        "details": details,
    }
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


async def test_enrich_writes_no_translations_even_if_a_decision_carries_some():
    """Translations are fontem-translator's (events); this executor writes
    the matching features only."""
    driver, captured = _capturing_driver()
    d = _decision(details={
        "translations": {"en": "Hello", "fr": "Bonjour"}, "source_lang": "de",
        "embedding": [0.1] * 768, "embedding_encoder": "labse@1.0.0-836121a",
    })
    await actions._enrich(driver, "neo4j", decision=d)
    props = captured["params"]["props"]
    assert set(props) == {"name_embedding", "name_embedding_encoder", "name_embedding_dim"}


async def test_nothing_to_embed_writes_nothing():
    driver, captured = _capturing_driver()
    await actions._enrich(driver, "neo4j", decision=_decision())
    assert "params" not in captured
