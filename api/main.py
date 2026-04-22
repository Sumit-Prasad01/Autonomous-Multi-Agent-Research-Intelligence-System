from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.router import router
from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.database.database import init_db
from src.research_intelligence_system.services.redis_service import get_redis
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)
_LOOP  = "uvloop" if sys.platform != "win32" else "asyncio"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up …")
    await init_db()
    await get_redis()
    # pre-warm embedding model
    from src.research_intelligence_system.rag.vector_store import _store
    asyncio.get_event_loop().run_in_executor(None, lambda: _store.embeddings)
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Intelligence API",
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)   # compress responses > 1KB

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing ────────────────────────────────────────────────────────
    @app.middleware("http")
    async def _timing(request: Request, call_next):
        t0  = time.perf_counter()
        res = await call_next(request)
        ms  = round((time.perf_counter() - t0) * 1000)
        res.headers["X-Response-Time-Ms"] = str(ms)
        logger.info(f"{request.method} {request.url.path} → {res.status_code} ({ms}ms)")
        return res

    # ── Global error handler ──────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _err(request: Request, exc: Exception):
        logger.exception(f"Unhandled: {request.url.path}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        loop=_LOOP,
        http="httptools",
    )