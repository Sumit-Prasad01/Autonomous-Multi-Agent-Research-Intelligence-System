from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.constants import ACCESS_TTL, ALGORITHM
from src.research_intelligence_system.database.models import User
from src.research_intelligence_system.services.redis_service import (
    blocklist_token, is_token_blocked
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password ──────────────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────
def create_token(user_id: str, email: str, username: str) -> str:
    jti = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    payload = {
        "sub":      user_id,
        "email":    email,
        "username": username,
        "jti":      jti,
        "iat":      now,
        "exp":      now + timedelta(minutes=ACCESS_TTL),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


async def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        if await is_token_blocked(payload["jti"]):
            return None
        return payload
    except JWTError:
        return None


# ── Register ──────────────────────────────────────────────────────────────────
async def register_user(
    db: AsyncSession, email: str, username: str, password: str
) -> User:
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        raise ValueError("Email already registered.")

    existing_username = await db.scalar(select(User).where(User.username == username))
    if existing_username:
        raise ValueError("Username already taken.")

    user = User(email=email, username=username, password=hash_password(password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info(f"Registered user {email}")
    return user


# ── Login ─────────────────────────────────────────────────────────────────────
async def login_user(
    db: AsyncSession, email: str, password: str
) -> Optional[str]:
    user = await db.scalar(
        select(User).where(User.email == email, User.is_active == True)
    )
    if not user or not verify_password(password, user.password):
        return None
    return create_token(str(user.id), user.email, user.username)


# ── Logout ────────────────────────────────────────────────────────────────────
async def logout_user(token: str) -> None:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_exp": False},
        )
        exp = payload.get("exp", 0)
        ttl = max(0, exp - int(datetime.now(timezone.utc).timestamp()))
        await blocklist_token(payload["jti"], ttl)
    except JWTError:
        pass