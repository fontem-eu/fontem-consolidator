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
"""
import pathlib


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


def test_nothing_writes_a_same_as_edge():
    """A :SAME_AS edge in Neo4j is a second copy of a fact Virtuoso
    already holds, and nothing in Neo4j follows it. Candidates and
    corrections are workflow and stay."""
    offenders = []
    for path, text in _sources():
        # The migration that DELETES the legacy edges necessarily names
        # the type it is removing.
        if path.name == "migrations.py":
            continue
        for line in text.splitlines():
            if "SAME_AS_CANDIDATE" in line or "NOT_SAME_AS" in line:
                continue
            if "[r:SAME_AS]" in line or "[:SAME_AS]" in line:
                offenders.append(f"{path}: {line.strip()}")
    assert not offenders, offenders
