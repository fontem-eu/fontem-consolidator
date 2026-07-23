"""Load Company / Authority / Contract nodes as Entity objects."""

from neo4j import AsyncDriver

from src.consolidator.rules.base import Entity

_ID_KEY_BY_TYPE: dict[str, str] = {
    "Company": "gmr_id",
    "Authority": "authority_id",
    "Contract": "ted_notice_id",
}


def id_key_for(entity_type: str) -> str:
    """Neo4j property that uniquely keys an entity of this type.

    Authority is the safe fallback for unknown labels — it matches the
    engine's default entity_type handling and keeps callers (e.g. the
    re-consolidation sweeper) from having to duplicate this mapping.
    """
    return _ID_KEY_BY_TYPE.get(entity_type, "authority_id")


async def load(
    driver: AsyncDriver,
    database: str,
    *,
    entity_type: str,
    entity_id: str,
) -> Entity | None:
    id_key = id_key_for(entity_type)
    async with driver.session(database=database) as session:
        result = await session.run(
            f"MATCH (n:{entity_type} {{{id_key}: $id}}) RETURN n",
            id=entity_id,
        )
        record = await result.single()
        if record is None:
            return None
        props = dict(record["n"])
    return Entity(entity_type=entity_type, id=entity_id, properties=props)
