from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import settings

# Lazy singleton — the driver is shared across rules; cleared on shutdown.
# Lowercase name is intentional (mutable cache, not a constant).
_driver: AsyncDriver | None = None  # pylint: disable=invalid-name


async def get_driver() -> AsyncDriver:
    global _driver  # pylint: disable=global-statement
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_driver() -> None:
    global _driver  # pylint: disable=global-statement
    if _driver is not None:
        await _driver.close()
        _driver = None
