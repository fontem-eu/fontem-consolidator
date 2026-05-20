"""Tests for the trade-edge refresh side of the merge action.

Background
==========
The materialize_trade_edges ETL job builds CLIENT_OF / SUPPLIER_OF
summary edges (Authority-[:CLIENT_OF {contracts, total_eur}]->Company
and the reverse SUPPLIER_OF) from the AWARDED / AWARDED_TO graph.
The graph view renders these summary edges by default; the contracts
list runs straight off AWARDED.

Pre-fix, when the consolidator merged duplicate Authority/Company
nodes, apoc.refactor.mergeNodes correctly rewrote the AWARDED edges
to the canonical, but it did NOT recompute the materialized summary
edges. Result: the canonical's CLIENT_OF count drifted from its
post-merge AWARDED count, and the two views diverged. Concrete
sighting: an Authority showed 4 contracts in the graph view (via
4 stale CLIENT_OF edges) and 0 in the contracts list.

These tests pin the new behaviour: every merge calls
``_refresh_trade_edges`` for the canonical, which drops and
rebuilds CLIENT_OF / SUPPLIER_OF for the canonical's neighborhood.
"""
# protected-access: tests pin the post-merge trade-edge rebuild
# by calling actions._merge / ._refresh_trade_edges and reading
# the _REFRESH_*_TRADE_EDGES cypher constants. Those are
# package-internal by design and the cleanest place to lock
# down the behaviour is the test that drives them.
# pylint: disable=protected-access
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.consolidator import actions
from src.consolidator.rules.base import Candidate, Decision, Entity


pytestmark = pytest.mark.asyncio


def _merge_decision(label: str = "Authority") -> Decision:
    return Decision(
        rule_name="exact_lei_match" if label == "Company" else "exact_authority_id_match",
        action="merge",
        source_id="CANONICAL-1",
        target_id="DUP-1",
        confidence=0.99,
        entity_type=label,
        details={},
    )


def _capturing_driver():
    """AsyncDriver stub that captures every cypher+params and lets
    us script the response of the existence check ("does the
    canonical+dup pair still exist?")."""
    captured: list[dict] = []

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    # The merge action's first .run() is the existence check; subsequent
    # runs are mergeNodes + trade-edge refresh. The existence check is
    # awaited and consumed via .single(); we have to fake that shape.
    async def _run(cypher, **params):
        captured.append({"cypher": cypher, "params": params})
        result = MagicMock()
        # First call is the existence check — return both nodes present.
        if "RETURN canonical, dup" in cypher and "MATCH (canonical:" in cypher:
            result.single = AsyncMock(return_value={"canonical": "x", "dup": "y"})
        else:
            result.single = AsyncMock(return_value=None)
        return result
    session.run = _run

    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver, captured


async def test_merge_authority_runs_trade_edge_refresh():
    """After merging two Authority nodes, the action must dispatch the
    Authority-flavored CLIENT_OF/SUPPLIER_OF refresh."""
    driver, captured = _capturing_driver()
    candidate = Candidate(entity=Entity("Authority", "DUP-1", {}), context={})
    await actions._merge(
        driver, "neo4j",
        decision=_merge_decision("Authority"),
        entity=Entity("Authority", "CANONICAL-1", {}),
        candidate=candidate,
    )
    # The third call is the trade-edge refresh (after existence check + mergeNodes).
    refresh = captured[-1]
    assert "MATCH (canonical:Authority {authority_id: $canonical_id})" in refresh["cypher"]
    assert "[:AWARDED]->(ct:Contract)-[:AWARDED_TO]->(c:Company)" in refresh["cypher"]
    assert "CREATE (canonical)-[:CLIENT_OF" in refresh["cypher"]
    assert "CREATE (c)-[:SUPPLIER_OF" in refresh["cypher"]
    assert refresh["params"] == {"canonical_id": "CANONICAL-1"}


async def test_merge_company_runs_trade_edge_refresh_inverted():
    """For Company canonicals, the refresh starts from the Company end
    so AWARDED_TO is the entry point. The summary edges still come out
    Authority -> Company (CLIENT_OF) and Company -> Authority
    (SUPPLIER_OF) — that direction is invariant."""
    driver, captured = _capturing_driver()
    candidate = Candidate(entity=Entity("Company", "DUP-1", {}), context={})
    await actions._merge(
        driver, "neo4j",
        decision=_merge_decision("Company"),
        entity=Entity("Company", "CANONICAL-1", {}),
        candidate=candidate,
    )
    refresh = captured[-1]
    assert "MATCH (canonical:Company {gmr_id: $canonical_id})" in refresh["cypher"]
    chain = "MATCH (a:Authority)-[:AWARDED]->(ct:Contract)-[:AWARDED_TO]->(canonical)"
    assert chain in refresh["cypher"]
    assert "CREATE (a)-[:CLIENT_OF" in refresh["cypher"]
    assert "CREATE (canonical)-[:SUPPLIER_OF" in refresh["cypher"]


async def test_merge_skips_trade_refresh_when_one_side_already_gone():
    """If the canonical or duplicate has been deleted between decision
    and execution (race with another rule), `_merge` returns early.
    The trade refresh must NOT run because there's nothing to refresh
    for — wasted writes mask a real bug."""
    driver, captured = _capturing_driver()

    # Override: existence check returns no record.
    session = driver.session.return_value
    async def _run_empty(cypher, **params):
        captured.append({"cypher": cypher, "params": params})
        result = MagicMock()
        result.single = AsyncMock(return_value=None)
        return result
    session.run = _run_empty

    candidate = Candidate(entity=Entity("Authority", "DUP-1", {}), context={})
    await actions._merge(
        driver, "neo4j",
        decision=_merge_decision("Authority"),
        entity=Entity("Authority", "CANONICAL-1", {}),
        candidate=candidate,
    )
    # Only the existence check fired; no merge, no refresh.
    assert len(captured) == 1
    assert "RETURN canonical, dup" in captured[0]["cypher"]


async def test_refresh_helper_is_idempotent_at_cypher_level():
    """The refresh starts with `OPTIONAL MATCH … DELETE` so a second
    invocation on the same canonical produces the same end state.
    Pin this by reading the cypher: the deletes must come BEFORE the
    creates."""
    auth_cypher = actions._REFRESH_AUTHORITY_TRADE_EDGES
    company_cypher = actions._REFRESH_COMPANY_TRADE_EDGES
    for cypher in (auth_cypher, company_cypher):
        delete_idx = cypher.index("DELETE")
        create_idx = cypher.index("CREATE")
        assert delete_idx < create_idx, (
            "trade-edge refresh must DELETE existing summary edges "
            "before CREATEing fresh ones, otherwise re-running the "
            "refresh duplicates them."
        )


async def test_refresh_helper_no_op_for_contract_label():
    """Trade edges aggregate (Authority, Company) pairs. Contract
    merges shouldn't try to dispatch them — no Cypher should fire."""
    session = MagicMock()
    session.run = AsyncMock()
    await actions._refresh_trade_edges(session, "Contract", "TED-123")
    session.run.assert_not_called()
