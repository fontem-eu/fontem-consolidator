"""A merge must not leave the pair's SAME_AS edge as a self-loop.

apoc.refactor.mergeNodes folds the relationships of both nodes onto the
survivor. The SAME_AS edge that justified the merge runs BETWEEN the two,
so by default it survives as a self-loop: the node now asserts it is the
same entity as itself. Verified against prod APOC — the identical merge
produces 1 self-loop with the default and 0 with produceSelfRel:false.

571 of these had accumulated by 2026-09-02 and are the standing
refs.sameas_no_selfloop block-tier failure. They also corrupt anything
that walks SAME_AS to build clusters, since every merged node becomes its
own neighbour.

The automatic path in actions._merge is covered here, along with a sweep
that catches any new call site appearing without the flag.

The review route no longer merges at all: approving a candidate writes a
:SAME_AS edge and leaves both nodes standing, because an assertion has to
remain correctable and :NOT_SAME_AS can only undo something that still
exists. That invariant is guarded below — if a merge ever reappears in
the review path, the correction endpoint silently stops working for
every pair it touches.
"""
import pathlib
import re

from src.consolidator.neo4j.migrations import BACKFILL_CYPHER

_MERGE_CALL = re.compile(
    r"apoc\.refactor\.mergeNodes\(.*?\{\{(.*?)\}\}",
    re.DOTALL,
)


def _merge_configs(path: str) -> list[str]:
    src = pathlib.Path(path).read_text(encoding="utf-8")
    # The Cypher is inside an f-string, so braces are doubled.
    return [m.group(1) for m in _MERGE_CALL.finditer(src)]


def test_actions_merge_disables_self_rel():
    configs = _merge_configs("src/consolidator/actions.py")
    assert configs, "no mergeNodes call found in actions.py"
    for cfg in configs:
        assert "produceSelfRel: false" in cfg


def test_review_approval_does_not_delete_a_node():
    """Approving a candidate must assert, not collapse.

    A correction (:NOT_SAME_AS + RetractSameAs) withdraws a published
    equivalence, which requires both nodes to still exist. If approval
    merged them, every approved pair would be permanently uncorrectable
    — and approval is the path a human explicitly signed off on, so it
    is the worst place to lose that ability.
    """
    src = pathlib.Path("src/api/routes/candidates.py").read_text(encoding="utf-8")
    assert "apoc.refactor.mergeNodes" not in src, (
        "the review route merges nodes again; approval must write a "
        ":SAME_AS edge and keep both nodes so corrections stay possible"
    )
    assert "assert_same_as" in src


def test_every_merge_call_site_is_covered():
    """Guards against a third copy appearing without the flag — the fix
    is only as good as its least-updated call site."""
    missing = []
    for path in pathlib.Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "apoc.refactor.mergeNodes" not in text:
            continue
        for cfg in _merge_configs(str(path)):
            if "produceSelfRel: false" not in cfg:
                missing.append(str(path))
    assert not missing, f"mergeNodes without produceSelfRel:false in {missing}"


def test_backfill_removes_existing_self_loops():
    """The flag stops new ones; the accumulated 571 still need deleting."""
    joined = " ".join(BACKFILL_CYPHER)
    assert "SAME_AS" in joined and "DELETE" in joined
