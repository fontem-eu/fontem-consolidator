"""Backfilling the :SAME_AS edges Neo4j never got.

The sink dropped every AssertSameAs for the period it had no renderer,
so Virtuoso holds 39,475 owl:sameAs and Neo4j holds none. The read path
follows the Neo4j edge now, so the gap has to be closed.

The dangerous mistake here is asserting a pair an operator has since
separated: 47,640 assertions were emitted and 7,872 retracted, and a
backfill driven by history rather than by current state would put every
one of those back and leave the graph contradicting its own
:NOT_SAME_AS edges. Most of what follows pins that it doesn't.
"""
# Fake emitters mirror emit_assert_same_as_many's signature exactly;
# *a/**k are the protocol shape, not something a stub reads.
# pylint: disable=protected-access,unused-argument
from src.consolidator import backfill_same_as as bf


def test_only_approved_candidates_are_asserted():
    """A pending proposal is a guess. 304,702 of them exist against
    39,193 approved, so asserting the wrong status would merge roughly
    eight times too much."""
    assert "status: 'approved'" in bf._FIND
    assert "SAME_AS_CANDIDATE" in bf._FIND


def test_a_corrected_pair_is_excluded():
    """The retraction case. An operator's :NOT_SAME_AS is the durable
    record that these two are NOT the same, and re-asserting over it
    trips refs.sameas_not_contradicted, which is BLOCK severity."""
    assert "NOT EXISTS { (a)-[:NOT_SAME_AS]-(b) }" in bf._FIND


def test_self_references_are_excluded():
    """A self-loop carries no information, trips
    refs.sameas_no_selfloop, and would make an identity-class traversal
    revisit its own start node. Guarded on both node identity and key
    equality, because two distinct nodes sharing a key is its own bug
    and must not become a self-edge either."""
    assert "elementId(a) <> elementId(b)" in bf._FIND
    assert "ak <> bk" in bf._FIND


def test_cross_label_pairs_are_excluded():
    """A Company and an Authority are not the same entity, and the sink
    skips such an assertion anyway — emitting it would just produce a
    warning per pair and no edge."""
    assert "labels(a)[0] = labels(b)[0]" in bf._FIND


def test_keyless_nodes_are_excluded():
    """The IRI is built by string concatenation, so a null key would
    silently produce '.../Company/null' and assert an equivalence
    against a node that does not exist."""
    assert "ak IS NOT NULL AND bk IS NOT NULL" in bf._FIND


def test_the_iri_is_built_from_the_label_and_key():
    """The sink parses IRIs back into (label, key) to find the nodes, so
    a malformed IRI here is an edge that silently never appears."""
    assert "'http://data.fontem.eu/id/' + label + '/' + ak" in bf._FIND
    assert "'http://data.fontem.eu/id/' + label + '/' + bk" in bf._FIND


def test_ordered_so_a_partial_run_is_legible():
    """The emit is batched and can stop part-way; a stable order means
    what landed is a prefix rather than an arbitrary subset."""
    assert "ORDER BY a_iri, b_iri" in bf._FIND


def test_dry_run_emits_nothing(monkeypatch):
    """--apply is the only thing that may write."""
    import asyncio  # pylint: disable=import-outside-toplevel

    calls = []

    async def _fake_find():
        return [{"a_iri": "x", "b_iri": "y", "confidence": 1.0,
                 "method": "m", "rule": None, "domain": "consolidation"}]

    async def _fake_emit(rows, *a, **k):
        calls.append(rows)
        return len(rows)

    monkeypatch.setattr(bf, "find_approved", _fake_find)
    monkeypatch.setattr(bf.eventlog, "emit_assert_same_as_many", _fake_emit)
    assert asyncio.run(bf.backfill(apply=False)) == 0
    assert not calls


def test_apply_emits_every_row_in_batches(monkeypatch):
    """And counts what actually landed: emit_assert_same_as_many
    swallows failures by design, so a silent short write must be
    visible in the return value rather than assumed."""
    import asyncio  # pylint: disable=import-outside-toplevel

    rows = [{"a_iri": f"a{i}", "b_iri": f"b{i}", "confidence": 1.0,
             "method": "m", "rule": None, "domain": "consolidation"}
            for i in range(250)]
    seen = []

    async def _fake_find():
        return rows

    async def _fake_emit(chunk, *a, **k):
        seen.append(len(chunk))
        return len(chunk)

    monkeypatch.setattr(bf, "find_approved", _fake_find)
    monkeypatch.setattr(bf.eventlog, "emit_assert_same_as_many", _fake_emit)
    assert asyncio.run(bf.backfill(apply=True, batch=100)) == 250
    assert seen == [100, 100, 50]


def test_a_short_emit_is_reported_not_swallowed(monkeypatch):
    """The event store dropping half a chunk must not read as success —
    the missing edges would be invisible until someone noticed a company
    page under-reporting."""
    import asyncio  # pylint: disable=import-outside-toplevel

    async def _fake_find():
        return [{"a_iri": f"a{i}", "b_iri": f"b{i}", "confidence": 1.0,
                 "method": "m", "rule": None, "domain": "consolidation"}
                for i in range(10)]

    async def _short_emit(chunk, *a, **k):
        return len(chunk) - 3

    monkeypatch.setattr(bf, "find_approved", _fake_find)
    monkeypatch.setattr(bf.eventlog, "emit_assert_same_as_many", _short_emit)
    assert asyncio.run(bf.backfill(apply=True, batch=10)) == 7
