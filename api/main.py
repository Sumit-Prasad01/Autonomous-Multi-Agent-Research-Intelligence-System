from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.router import router
from src.research_intelligence_system.rag.vector_store import _store
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# uvloop is not supported on Windows
_LOOP    = "uvloop" if sys.platform != "win32" else "asyncio"
_HTTP    = "httptools"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up vector store …")
    _ = _store.db
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research Intelligence API",
        description="Agentic RAG QA system — Groq/Llama + FAISS + BM25 + Tavily",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        t0  = time.perf_counter()
        res = await call_next(request)
        ms  = round((time.perf_counter() - t0) * 1000)
        res.headers["X-Response-Time-Ms"] = str(ms)
        logger.info(f"{request.method} {request.url.path} → {res.status_code} ({ms}ms)")
        return res

    @app.exception_handler(Exception)
    async def _global_error(request: Request, exc: Exception):
        logger.exception(f"Unhandled error on {request.url.path}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1, # Production on a real server with 16 GB+ RAM → workers=2 or workers=4 || Windows dev machine → always workers=1
        loop=_LOOP,
        http=_HTTP,
    )
