# gmr-consolidator

Data consolidation service for the GMR knowledge graph.

Applies a pipeline of deterministic + graph-based rules to resolve duplicate
`:Company` and `:Authority` nodes, with a full audit trail in Neo4j.

Singleton deployment in the `gmr` namespace — Neo4j is shared across all
environments, so one consolidator serves everything.

## Local

```
pip install -r requirements-dev.txt
pytest tests/unit/
uvicorn src.api.app:app --reload
```

## Endpoints

- `POST /consolidate/company/{gmr_id}`
- `POST /consolidate/authority/{authority_id}`
- `POST /consolidate/batch`
- `GET  /candidates?reviewed=false`
- `POST /candidates/{from}/{to}/decide`
- `GET  /decisions`
- `GET  /rules`
- `POST /webhooks/neo4j-trigger`
- `GET  /health`, `GET /metrics`

## Audit

Every rule outcome — auto-merge, link, flag, conflict, noop — and every
manual decision via `/candidates/.../decide` lands as a `:DecisionLog`
node linked to a `:ConsolidationRun` and `:RuleApplication`, plus the
existing `:MergeEvent` for executed merges.
