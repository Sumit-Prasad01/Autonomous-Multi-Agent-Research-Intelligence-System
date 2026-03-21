from __future__ import annotations

import asyncio
import os
import threading
import uuid
from typing import Dict

import aiofiles
from fastapi import (APIRouter, Depends, HTTPException, UploadFile, File)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from concurrent.futures import ThreadPoolExecutor
from api.schemas import (
    ChatCreateRequest, ChatMessageRequest, ChatUpdateRequest,
    LoginRequest, RegisterRequest,
)
from src.research_intelligence_system.constants import BASE_DIR, MAX_MB
from src.research_intelligence_system.database.chat_repository import (
    create_chat, delete_chat, get_chat, get_user_chats, update_chat_title
)
from src.research_intelligence_system.database.database import get_db, check_db_health
from src.research_intelligence_system.pipeline.paper_processing_pipeline import ingest_pdf
from src.research_intelligence_system.rag.retriever import invalidate_retriever_cache
from src.research_intelligence_system.services.auth_service import (
    decode_token, login_user, logout_user, register_user
)
from src.research_intelligence_system.services.chat_service import (
    delete_chat_session, load_chat_history, process_chat, stream_chat
)
from src.research_intelligence_system.services.redis_service import check_redis_health
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.services.redis_service import (
    check_redis_health, create_session, get_session, delete_session
)


logger   = get_logger(__name__)
router   = APIRouter(prefix="/api/v1", tags=["api"])
security = HTTPBearer()

_ingest_status: Dict[str, Dict] = {}
_ingest_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")

# ── Auth dependency ───────────────────────────────────────────────────────────
async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    payload = await decode_token(creds.credentials)
    if not payload:
        raise HTTPException(401, "Invalid or expired token.")
    return payload


# ── Auth ──────────────────────────────────────────────────────────────────────
@router.post("/auth/register", status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await register_user(db, req.email, req.username, req.password)
        return {"id": str(user.id), "email": user.email, "username": user.username}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/auth/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    token = await login_user(db, req.email, req.password)
    if not token:
        raise HTTPException(401, "Invalid credentials.")
    session_id = await create_session(token, req.email)
    return {"access_token": token, "token_type": "bearer", "session_id": session_id}


@router.post("/auth/logout")
async def logout(creds: HTTPAuthorizationCredentials = Depends(security)):
    await logout_user(creds.credentials)
    return {"message": "Logged out."}

# ── User Session ─────────────────────────────────────────────────────────────
@router.get("/auth/session/{session_id}")
async def restore_session(session_id: str):
    data = await get_session(session_id)
    if not data:
        raise HTTPException(401, "Session expired.")
    return data

@router.delete("/auth/session/{session_id}")
async def remove_session(session_id: str):
    await delete_session(session_id)
    return {"message": "Session deleted."}

# ── Chat CRUD ─────────────────────────────────────────────────────────────────
@router.post("/chats", status_code=201)
async def create_new_chat(
    req: ChatCreateRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await create_chat(db, user["sub"], req.title, req.llm_id, req.allow_search)
    return {"id": str(chat.id), "title": chat.title,
            "llm_id": chat.llm_id, "created_at": str(chat.created_at)}


@router.get("/chats")
async def list_chats(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chats = await get_user_chats(db, user["sub"])
    return [{"id": str(c.id), "title": c.title,
             "llm_id": c.llm_id, "updated_at": str(c.updated_at)}
            for c in chats]


@router.get("/chats/{chat_id}/history")
async def get_history(
    chat_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    return await load_chat_history(chat_id, db)


@router.patch("/chats/{chat_id}")
async def rename_chat(
    chat_id: str,
    req: ChatUpdateRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    await update_chat_title(db, chat_id, req.title)
    return {"message": "Updated."}


@router.delete("/chats/{chat_id}")
async def remove_chat(
    chat_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_chat(db, chat_id, user["sub"])
    if not ok:
        raise HTTPException(404, "Chat not found.")
    await delete_chat_session(chat_id)
    invalidate_retriever_cache(chat_id)
    return {"message": "Deleted."}


# ── Messaging ─────────────────────────────────────────────────────────────────
@router.post("/chats/{chat_id}/message")
async def send_message(
    chat_id: str,
    req: ChatMessageRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    return await process_chat(
        chat_id=chat_id,
        user_message=req.message,
        llm_id=req.llm_id,
        allow_search=req.allow_search,
        db=db,
    )


@router.post("/chats/{chat_id}/stream")
async def stream_message(
    chat_id: str,
    req: ChatMessageRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")

    async def _events():
        try:
            async for token in stream_chat(
                chat_id=chat_id,
                user_message=req.message,
                llm_id=req.llm_id,
                allow_search=req.allow_search,
                db=db,
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


# ── PDF upload ────────────────────────────────────────────────────────────────
def _run_ingest_sync(chat_id: str, file_path: str):
    """Sync wrapper — runs in threadpool, shares main process memory."""
    import asyncio
    _ingest_status[chat_id] = {"status": "processing", "error": None}
    try:
        logger.info(f"[INGEST] starting chat_id={chat_id}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ingest_pdf(chat_id=chat_id, pdf_path=file_path))
        _ingest_status[chat_id] = {"status": "ready", "error": None}
        logger.info(f"[INGEST] ready chat_id={chat_id}")
    except Exception as e:
        _ingest_status[chat_id] = {"status": "failed", "error": str(e)}
        logger.error(f"[INGEST] failed: {e}", exc_info=True)
    finally:
        loop.close()


@router.post("/chats/{chat_id}/upload")
async def upload_pdf(
    chat_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB} MB.")

    chat_dir  = os.path.join(BASE_DIR, chat_id)
    os.makedirs(chat_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    file_path = os.path.join(chat_dir, safe_name)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    _ingest_pool.submit(_run_ingest_sync, chat_id, file_path)

    return {"status": "processing", "filename": safe_name}


@router.get("/chats/{chat_id}/status")
async def ingest_status(
    chat_id: str,
    user: dict = Depends(get_current_user),
):
    return _ingest_status.get(chat_id, {"status": "unknown", "error": None})


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    db_ok    = await check_db_health()
    redis_ok = await check_redis_health()
    return {
        "status":   "ok" if db_ok and redis_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
        "redis":    "ok" if redis_ok else "unreachable",
    }
