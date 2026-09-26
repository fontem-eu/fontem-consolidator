"""Nothing in the consolidator merges nodes any more.

Merging a duplicate into its canonical was Neo4j's substitute for
reasoning it cannot do. It destroyed the losing node, baked an identity
decision irreversibly into the graph store, and duplicated a fact
Virtuoso already holds properly as owl:sameAs.

Identity now lives in Virtuoso alone; Neo4j keeps the graph and the
review workflow. These tests guard the boundary, because the failure
mode is silent: a merge reappearing anywhere would start deleting nodes
again, and the deleted node is the one a :NOT_SAME_AS correction would
need in order to undo the mistake.

The old worry here was `produceSelfRel` — a merge turned the SAME_AS
edge BETWEEN the pair into a self-loop on the survivor, and 571 had
accumulated by 2026-09-02. That whole class of bug is gone with the
merges.

:SAME_AS itself is back (fontem-neo4j-sink#148): the read path resolves
an identity class by traversing it, which is what nothing in Neo4j used
to do. Written by the SINK from AssertSameAs, never by the consolidator
— so it stays derivable from the event log — and never deleted on
startup, which is the trap the last test here guards.
"""
import pathlib

from src.consolidator.neo4j import migrations


def _sources():
    for path in pathlib.Path("src").rglob("*.py"):
        yield path, path.read_text(encoding="utf-8")


def test_nothing_merges_nodes():
    # Only actual call sites; the comments explaining why merging went
    # away are the point of keeping this readable.
    offenders = [
        str(p) for p, text in _sources()
        if "CALL apoc.refactor.mergeNodes" in text
    ]
    assert not offenders, (
        f"{offenders} still merge nodes; identity belongs in Virtuoso and a "
        "merged node cannot be restored by a :NOT_SAME_AS correction"
    )


def test_the_consolidator_never_writes_a_same_as_edge_itself():
    """:SAME_AS is back in Neo4j — the read path traverses it to resolve
    an identity class — but the consolidator still must not write it.

    It emits AssertSameAs; the neo4j sink renders the edge. That is what
    keeps the edge derivable from the event log, so a replay from seq 0
    reconstructs identity instead of leaving whatever a sweep happened
    to write. A consolidator that wrote the edge directly would also be
    writing it for pairs it has only PROPOSED, which is the distinction
    :SAME_AS_CANDIDATE exists to hold.
    """
    offenders = []
    for path, text in _sources():
        for line in text.splitlines():
            if "SAME_AS_CANDIDATE" in line or "NOT_SAME_AS" in line:
                continue
            if "[r:SAME_AS]" in line or "[:SAME_AS]" in line:
                offenders.append(f"{path}: {line.strip()}")
    assert not offenders, offenders


def test_startup_deletes_nothing():
    """The landmine. A startup migration used to end by deleting every
    :SAME_AS in the graph: correct while Neo4j held no equivalences, a
    silent wipe of identity once the sink wrote them. Startup now only
    ensures indexes; nothing it runs may delete."""
    for stmt in migrations.INDEX_CYPHER:
        assert "DELETE" not in stmt.upper(), (
            f"a startup migration deletes: {stmt.strip()}"
        )
