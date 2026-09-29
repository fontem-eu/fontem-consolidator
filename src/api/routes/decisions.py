"""GET /decisions — what the consolidator decided, newest first.

Read from Postgres (consolidation.decision_log, see audit.py): rule
decisions that changed something, and every human decision.
"""
from fastapi import APIRouter, Query

from src.consolidator import audit

router = APIRouter()


# One parameter per filter of the query string.
@router.get("/decisions")
async def list_decisions(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    rule_name: str | None = Query(default=None),
    decision_type: str | None = Query(default=None),
    since: str | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
    cursor: str | None = Query(default=None),
):
    rows = await audit.list_decisions(
        entity_type=entity_type, entity_id=entity_id, rule_name=rule_name,
        decision_type=decision_type, since=since, cursor=cursor, limit=limit,
    )
    return {
        "decisions": rows,
        "next_cursor": rows[-1]["decided_at"] if len(rows) == limit else None,
    }
