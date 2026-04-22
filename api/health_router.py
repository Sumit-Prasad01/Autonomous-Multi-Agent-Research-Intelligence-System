"""
health_router.py — Health check endpoint
GET /health
"""
from __future__ import annotations

from fastapi import APIRouter

from src.research_intelligence_system.database.database import check_db_health
from src.research_intelligence_system.services.redis_service import check_redis_health

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    db_ok    = await check_db_health()
    redis_ok = await check_redis_health()
    return {
        "status":   "ok" if db_ok and redis_ok else "degraded",
        "database": "ok" if db_ok    else "unreachable",
        "redis":    "ok" if redis_ok else "unreachable",
    }