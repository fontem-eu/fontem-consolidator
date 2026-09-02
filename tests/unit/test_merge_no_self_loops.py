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

Both merge paths are covered: the automatic one in actions._merge and the
manual review one in the candidates route. They merge the same way, so
they need the same flag, and a fix applied to only one of them would look
correct until somebody used the review queue.
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


def test_manual_review_merge_disables_self_rel():
    """The review queue merges through its own copy of this Cypher."""
    configs = _merge_configs("src/api/routes/candidates.py")
    assert configs, "no mergeNodes call found in candidates.py"
    for cfg in configs:
        assert "produceSelfRel: false" in cfg


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
