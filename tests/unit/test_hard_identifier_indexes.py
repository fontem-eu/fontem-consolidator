"""Every property a dedup rule matches on must be indexed.

An _ExactIdRule runs

    MATCH (c:Company) WHERE c.<id> = $value AND c.gmr_id <> $self_id

for every entity carrying that identifier. Unindexed, each one is a
NodeByLabelScan over the whole Company graph — 3,516,650 nodes in
shared — and a company with all three identifiers pays it three times.

Measured on shared 2026-09-07 before the indexes existed: Neo4j pinned
at 3,884m of a 4-core limit, sweeper at 6 companies/sec, trigger at
0.72 events/sec. A full Company sweep would have taken ~6.7 days and
the trigger's backlog ~113 days.

This is the same failure that capped the trigger at 0.06 events/sec
before consolidationrun_run_id was added, so the test is written
against the rules rather than against a hardcoded list: a new
_ExactIdRule subclass with an unindexed id_property fails here instead
of silently halving throughput in production.
"""
from src.consolidator.neo4j.migrations import INDEX_CYPHER
from src.consolidator.rules.company import exact_identifiers


def _indexed_company_properties() -> set[str]:
    props: set[str] = set()
    for stmt in INDEX_CYPHER:
        if "(c:Company)" not in stmt or "ON (" not in stmt:
            continue
        inner = stmt.split("ON (", 1)[1].rsplit(")", 1)[0]
        for part in inner.split(","):
            part = part.strip()
            if part.startswith("c."):
                props.add(part[2:])
    return props


def _exact_id_rule_properties() -> set[str]:
    found = set()
    for name in dir(exact_identifiers):
        obj = getattr(exact_identifiers, name)
        prop = getattr(obj, "id_property", None)
        if isinstance(obj, type) and isinstance(prop, str):
            found.add(prop)
    return found


def test_every_hard_identifier_rule_matches_on_an_indexed_property():
    """The rules that auto-merge are the ones that run on every entity,
    so an unindexed one costs a full label scan per consolidation."""
    rule_props = _exact_id_rule_properties()
    assert rule_props, "found no _ExactIdRule subclasses to check"
    missing = sorted(rule_props - _indexed_company_properties())
    assert not missing, (
        f"hard-identifier rules match on unindexed Company properties: "
        f"{missing}. Each costs a NodeByLabelScan over the whole Company "
        f"graph per consolidation — add an index in migrations.INDEX_CYPHER."
    )


def test_the_three_known_identifiers_are_covered():
    """Pins the specific set, so removing an index is a failure even if
    the rule is removed in the same change by accident."""
    indexed = _indexed_company_properties()
    for prop in ("lei", "cik", "vat"):
        assert prop in indexed, f"Company.{prop} lost its index"
