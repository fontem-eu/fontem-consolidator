"""On-demand matching-quality report against a live graph.

Run it against shared (safe) to get the *real* numbers the CI test can't:
precision from LEI homonyms, the worst homonym offenders, the recall-gap
population (same-entity VAT pairs with name drift), and a continuous
fidelity check of the clean() replica vs the materialised name_clean (so an
apoc upgrade that changes normalisation is caught).

  NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD  — connection (env)
  usage:  python -m eval.live_report
"""
from __future__ import annotations

import os

from neo4j import GraphDatabase

from src.consolidator.eval.clean import clean

_PRECISION = """
MATCH (c:Company)
WHERE c.lei IS NOT NULL AND c.name_clean IS NOT NULL AND c.country IS NOT NULL
WITH c.name_clean AS nc, c.country AS ct, count(DISTINCT c.lei) AS lc
WITH count(*) AS groups,
     sum(CASE WHEN lc>1 THEN 1 ELSE 0 END) AS homonym_groups,
     sum(CASE WHEN lc>1 THEN lc ELSE 0 END) AS entities_at_risk
RETURN groups, homonym_groups, entities_at_risk
"""

_WORST = """
MATCH (c:Company)
WHERE c.lei IS NOT NULL AND c.name_clean IS NOT NULL AND c.country IS NOT NULL
WITH c.name_clean AS nc, c.country AS ct, count(DISTINCT c.lei) AS lc
WHERE lc > 1 RETURN nc AS name_clean, ct AS country, lc AS distinct_leis
ORDER BY lc DESC LIMIT 8
"""

_RECALL_GAP = """
MATCH (c:Company) WHERE c.vat IS NOT NULL AND c.name_clean IS NOT NULL
WITH c.vat AS vat, count(DISTINCT c.name_clean) AS names
WHERE names > 1 RETURN count(*) AS same_entity_name_drift_pairs
"""

_FIDELITY = """
MATCH (c:Company) WHERE c.name IS NOT NULL AND c.name_clean IS NOT NULL
RETURN c.name AS name, c.name_clean AS name_clean LIMIT $n
"""


def _run(session, query, **params):
    return [dict(r) for r in session.run(query, **params)]


def report(driver, sample: int = 2000) -> dict:
    out = {}
    with driver.session() as s:
        p = _run(s, _PRECISION)[0]
        groups, hom = p["groups"], p["homonym_groups"]
        out["precision"] = {
            **p,
            "group_precision": round((groups - hom) / groups, 4) if groups else 1.0,
        }
        out["worst_homonyms"] = _run(s, _WORST)
        out["recall_gap"] = _run(s, _RECALL_GAP)[0]
        rows = _run(s, _FIDELITY, n=sample)
    ok = sum(1 for r in rows if clean(r["name"]) == r["name_clean"])
    out["replica_fidelity"] = {
        "checked": len(rows), "match": ok,
        "pct": round(100 * ok / len(rows), 2) if rows else None,
    }
    return out


def main() -> None:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    try:
        rep = report(driver)
    finally:
        driver.close()
    pr = rep["precision"]
    print(f"PRECISION (LEI ground truth): {pr['group_precision']:.4f}")
    print(f"  {pr['homonym_groups']} homonym groups / {pr['entities_at_risk']} "
          f"entities a blind name_country backfill would over-merge")
    for w in rep["worst_homonyms"]:
        print(f"    {w['name_clean']!r} ({w['country']}): {w['distinct_leis']} real companies")
    print(f"RECALL gap (same-entity name drift, VAT-confirmed): "
          f"{rep['recall_gap']['same_entity_name_drift_pairs']} pairs")
    f = rep["replica_fidelity"]
    print(f"REPLICA fidelity vs apoc.text.clean: {f['match']}/{f['checked']} = {f['pct']}%")


if __name__ == "__main__":
    main()
