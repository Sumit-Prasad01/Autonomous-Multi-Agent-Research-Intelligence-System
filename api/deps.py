"""
deps.py — Shared FastAPI dependencies
"""
from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.research_intelligence_system.services.auth_service import decode_token

security = HTTPBearer()


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Decode JWT and return payload. Raises 401 if invalid."""
    payload = await decode_token(creds.credentials)
    if not payload:
        raise HTTPException(401, "Invalid or expired token.")
    return payload