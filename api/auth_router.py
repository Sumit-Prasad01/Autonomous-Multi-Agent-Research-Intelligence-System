"""
auth_router.py — Authentication + session endpoints
POST /auth/register
POST /auth/login
POST /auth/logout
GET  /auth/session/{session_id}
DELETE /auth/session/{session_id}
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import LoginRequest, RegisterRequest
from src.research_intelligence_system.database.database import get_db
from src.research_intelligence_system.services.auth_service import (
    decode_token, login_user, logout_user, register_user,
)
from src.research_intelligence_system.services.redis_service import (
    create_session, delete_session, get_session,
)

router   = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


@router.post("/register", status_code=201)
async def register(
    req: RegisterRequest,
    db:  AsyncSession = Depends(get_db),
):
    try:
        user = await register_user(db, req.email, req.username, req.password)
        return {"id": str(user.id), "email": user.email, "username": user.username}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/login")
async def login(
    req: LoginRequest,
    db:  AsyncSession = Depends(get_db),
):
    token = await login_user(db, req.email, req.password)
    if not token:
        raise HTTPException(401, "Invalid credentials.")
    session_id = await create_session(token, req.email)
    return {"access_token": token, "token_type": "bearer", "session_id": session_id}


@router.post("/logout")
async def logout(
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    await logout_user(creds.credentials)
    return {"message": "Logged out."}


@router.get("/session/{session_id}")
async def restore_session(session_id: str):
    data = await get_session(session_id)
    if not data:
        raise HTTPException(401, "Session expired.")
    return data


@router.delete("/session/{session_id}")
async def remove_session(session_id: str):
    await delete_session(session_id)
    return {"message": "Session deleted."}