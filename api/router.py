"""
router.py — Main API router
Registers all sub-routers under /api/v1 prefix.
Import this in main.py and include it in the FastAPI app.
"""
from __future__ import annotations

from fastapi import APIRouter

from api.auth_router   import router as auth_router
from api.chat_router   import router as chat_router
from api.paper_router  import router as paper_router
from api.health_router import router as health_router

# single entry point — all routers mounted here
router = APIRouter(prefix="/api/v1")

router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(paper_router)
router.include_router(health_router)