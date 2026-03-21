from __future__ import annotations

import uuid
import json
import threading
from typing import List, Optional

import redis.asyncio as aioredis

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.constants import MAX_MEMORY_MSGS, MSG_TTL
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_redis: Optional[aioredis.Redis] = None
_redis_lock = threading.Lock()


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        with _redis_lock:
            if _redis is None:
                _redis = await aioredis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                logger.info("Redis connected.")
    return _redis


async def check_redis_health() -> bool:
    try:
        r = await get_redis()
        return await r.ping()
    except Exception:
        return False


# ── Chat memory ───────────────────────────────────────────────────────────────
def _key(chat_id: str) -> str:
    return f"chat_memory:{chat_id}"


async def push_message(chat_id: str, role: str, content: str) -> None:
    """Append message and keep only last MAX_MEMORY_MSGS — uses pipeline for 1 round trip."""
    r   = await get_redis()
    key = _key(chat_id)
    msg = json.dumps({"role": role, "content": content})
    async with r.pipeline(transaction=True) as pipe:
        await pipe.rpush(key, msg)
        await pipe.ltrim(key, -MAX_MEMORY_MSGS, -1)
        await pipe.expire(key, MSG_TTL)
        await pipe.execute()


async def get_memory(chat_id: str) -> List[dict]:
    r   = await get_redis()
    raw = await r.lrange(_key(chat_id), 0, -1)
    return [json.loads(m) for m in raw]


async def clear_memory(chat_id: str) -> None:
    r = await get_redis()
    await r.delete(_key(chat_id))


# ── JWT blocklist ─────────────────────────────────────────────────────────────
async def blocklist_token(jti: str, ttl_seconds: int) -> None:
    r = await get_redis()
    await r.setex(f"blocklist:{jti}", ttl_seconds, "1")


async def is_token_blocked(jti: str) -> bool:
    r = await get_redis()
    return await r.exists(f"blocklist:{jti}") == 1

#  ── User Session ─────────────────────────────────────────────────────────────

async def create_session(token: str, user: str) -> str:
    session_id = str(uuid.uuid4())
    r = await get_redis()
    await r.setex(f"session:{session_id}", 86400,
                  json.dumps({"token": token, "user": user}))
    return session_id

async def get_session(session_id: str) -> Optional[dict]:
    r = await get_redis()
    raw = await r.get(f"session:{session_id}")
    return json.loads(raw) if raw else None

async def delete_session(session_id: str) -> None:
    r = await get_redis()
    await r.delete(f"session:{session_id}")