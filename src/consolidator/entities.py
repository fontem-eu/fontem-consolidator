"""Load Company / Authority nodes as Entity objects."""

from neo4j import AsyncDriver

from src.consolidator.rules.base import Entity


async def load(
    driver: AsyncDriver,
    database: str,
    *,
    entity_type: str,
    entity_id: str,
) -> Entity | None:
    id_key = "gmr_id" if entity_type == "Company" else "authority_id"
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
