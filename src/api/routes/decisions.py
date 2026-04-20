from fastapi import APIRouter, Query

from src.config import settings
from src.consolidator.neo4j.client import get_driver

router = APIRouter()


@router.get("/decisions")
async def list_decisions(
    entity_type: str | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    rule_name: str | None = Query(default=None),
    decision_type: str | None = Query(default=None),
    since: str | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
    cursor: str | None = Query(default=None),
):
    where = []
    params: dict = {"limit": limit}
    if entity_type:
        where.append("d.entity_type = $entity_type")
        params["entity_type"] = entity_type
    if entity_id:
        where.append("(d.source_id = $entity_id OR d.target_id = $entity_id)")
        params["entity_id"] = entity_id
    if rule_name:
        where.append("d.rule_name = $rule_name")
        params["rule_name"] = rule_name
    if decision_type:
        where.append("d.decision_type = $decision_type")
        params["decision_type"] = decision_type
    if since:
        where.append("d.decided_at >= $since")
        params["since"] = since
    if cursor:
        where.append("d.decided_at < $cursor")
        params["cursor"] = cursor

    driver = await get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            f"""
            MATCH (d:DecisionLog)
            {"WHERE " + " AND ".join(where) if where else ""}
            RETURN d ORDER BY d.decided_at DESC LIMIT $limit
            """,
            **params,
        )
        rows = [dict(r["d"]) async for r in result]
    return {
        "decisions": rows,
        "next_cursor": rows[-1]["decided_at"] if len(rows) == limit else None,
    }
