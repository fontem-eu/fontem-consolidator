"""GDS rules — unit tests with a mocked driver. We assert the rules are well-formed
and degrade gracefully when GDS is absent. Real GDS behavior is exercised by
integration tests against a testcontainer."""

from unittest.mock import AsyncMock, patch

import pytest

from src.consolidator.rules.base import Candidate, Entity
from src.consolidator.rules.gds.node_similarity import (
    GdsNodeSimilarityAuthority,
    GdsNodeSimilarityCompany,
)
from src.consolidator.rules.gds.wcc_collapse import (
    GdsSameAsClusterCollapseAuthority,
    GdsSameAsClusterCollapseCompany,
)


@pytest.mark.asyncio
async def test_gds_node_similarity_company_applies_only_to_company():
    rule = GdsNodeSimilarityCompany()
    assert await rule.applies(Entity("Company", "A", {})) is True
    assert await rule.applies(Entity("Authority", "A", {})) is False


@pytest.mark.asyncio
async def test_gds_node_similarity_authority_applies_only_to_authority():
    rule = GdsNodeSimilarityAuthority()
    assert await rule.applies(Entity("Authority", "A", {})) is True
    assert await rule.applies(Entity("Company", "A", {})) is False


@pytest.mark.asyncio
async def test_gds_node_similarity_empty_when_gds_absent():
    rule = GdsNodeSimilarityCompany()
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock()), \
         patch("src.consolidator.rules.gds.node_similarity.gds_available", AsyncMock(return_value=False)):
        candidates = await rule.find_candidates(Entity("Company", "A", {}))
    assert candidates == []


@pytest.mark.asyncio
async def test_gds_node_similarity_resolve_emits_flag():
    rule = GdsNodeSimilarityCompany()
    entity = Entity("Company", "A", {})
    candidate = Candidate(entity=Entity("Company", "B", {}), context={"jaccard": 0.82})
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "flag"
    assert decision.confidence == pytest.approx(0.82)
    assert decision.details["method"] == "gds_node_similarity"


@pytest.mark.asyncio
async def test_gds_wcc_collapse_empty_when_gds_absent():
    rule = GdsSameAsClusterCollapseCompany()
    with patch("src.consolidator.neo4j.client.get_driver", AsyncMock()), \
         patch("src.consolidator.rules.gds.wcc_collapse.gds_available", AsyncMock(return_value=False)):
        candidates = await rule.find_candidates(Entity("Company", "A", {}))
    assert candidates == []


@pytest.mark.asyncio
async def test_gds_wcc_collapse_resolve_emits_merge():
    rule = GdsSameAsClusterCollapseAuthority()
    entity = Entity("Authority", "A", {})
    candidate = Candidate(entity=Entity("Authority", "B", {}), context={})
    decision = await rule.resolve(entity, candidate)
    assert decision.action == "merge"
    assert decision.confidence == 1.0
    assert decision.details["method"] == "gds_wcc_same_as_cluster"
