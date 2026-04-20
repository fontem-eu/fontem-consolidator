from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.routes.consolidate import router as consolidate_router
from src.config import settings
from src.consolidator.neo4j import client as neo4j_client
from src.consolidator.neo4j import migrations
from src.consolidator.rules.loader import load_all as load_rules


@asynccontextmanager
async def lifespan(_: FastAPI):
    driver = await neo4j_client.get_driver()
    await migrations.apply(driver, settings.neo4j_database)
    load_rules()
    logger.info("consolidator: startup complete (auto_merge={})", settings.auto_merge_enabled)
    yield
    await neo4j_client.close_driver()


app = FastAPI(title="gmr-consolidator", version="0.1.0", lifespan=lifespan)
app.include_router(consolidate_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/rules")
async def rules():
    from src.consolidator.rules.registry import list_rules

    return [r.describe() for r in list_rules()]
