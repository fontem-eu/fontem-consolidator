"""Every property the audit chain looks a node up by must be indexed.

audit.record_decision runs once per decision — dozens of times per
consolidated entity — and each call MATCHes a ConsolidationRun by
run_id. When that property is unindexed the MATCH degrades to a
NodeByLabelScan over every run ever recorded, which in prod meant
4,775,388 nodes and 2.9s to fetch a single node.

The failure mode is nasty precisely because nothing errors: the query
is correct, the results are right, and the pipeline just gets slower
every day as the scanned label grows. This test pins the index so the
regression cannot come back silently.
"""
import pathlib
import re

from src.consolidator.neo4j.migrations import INDEX_CYPHER


def _indexed_properties() -> set[tuple[str, str]]:
    """(label, property) pairs covered by a single-property index."""
    out = set()
    for stmt in INDEX_CYPHER:
        m = re.search(r"FOR \(\w+:(\w+)\) ON \(([^)]*)\)", stmt)
        if not m:
            continue
        label, props = m.group(1), m.group(2)
        parts = [p.strip().split(".")[-1] for p in props.split(",")]
        # Only the leading property of a composite index can serve a
        # lookup on that property alone.
        if parts:
            out.add((label, parts[0]))
    return out


def test_consolidation_run_id_is_indexed():
    """The lookup on the hot path of every single decision."""
    assert ("ConsolidationRun", "run_id") in _indexed_properties()


def test_audit_lookup_properties_are_all_indexed():
    """Guards the whole audit chain, not just today's offender —
    a new MATCH-by-id in audit.py should fail here, not in prod."""
    audit_src = pathlib.Path("src/consolidator/audit.py").read_text(encoding="utf-8")
    looked_up = set(re.findall(r"MATCH \(\w+:(\w+) \{(\w+):", audit_src))
    indexed = _indexed_properties()
    missing = {(lbl, prop) for lbl, prop in looked_up if (lbl, prop) not in indexed}
    assert not missing, f"audit.py looks up unindexed properties: {sorted(missing)}"
