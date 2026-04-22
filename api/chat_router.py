"""
chat_router.py — Chat CRUD + messaging endpoints
POST   /chats
GET    /chats
GET    /chats/{chat_id}/history
PATCH  /chats/{chat_id}
DELETE /chats/{chat_id}
POST   /chats/{chat_id}/message
POST   /chats/{chat_id}/stream
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from api.schemas import ChatCreateRequest, ChatMessageRequest, ChatUpdateRequest
from src.research_intelligence_system.database.chat_repository import (
    create_chat, delete_chat, get_chat, get_user_chats, update_chat_title,
)
from src.research_intelligence_system.database.database import get_db
from src.research_intelligence_system.rag.retriever import invalidate_retriever_cache
from src.research_intelligence_system.services.chat_service import (
    delete_chat_session, load_chat_history, process_chat, stream_chat,
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chats", tags=["chats"])


@router.post("", status_code=201)
async def create_new_chat(
    req:  ChatCreateRequest,
    user: dict = Depends(get_current_user),
    db:   AsyncSession = Depends(get_db),
):
    chat = await create_chat(db, user["sub"], req.title, req.llm_id, req.allow_search)
    return {
        "id":         str(chat.id),
        "title":      chat.title,
        "llm_id":     chat.llm_id,
        "created_at": str(chat.created_at),
    }


@router.get("")
async def list_chats(
    user: dict = Depends(get_current_user),
    db:   AsyncSession = Depends(get_db),
):
    chats = await get_user_chats(db, user["sub"])
    return [
        {"id": str(c.id), "title": c.title,
         "llm_id": c.llm_id, "updated_at": str(c.updated_at)}
        for c in chats
    ]


@router.get("/{chat_id}/history")
async def get_history(
    chat_id: str,
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    return await load_chat_history(chat_id, db)


@router.patch("/{chat_id}")
async def rename_chat(
    chat_id: str,
    req:     ChatUpdateRequest,
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    await update_chat_title(db, chat_id, req.title)
    return {"message": "Updated."}


@router.delete("/{chat_id}")
async def remove_chat(
    chat_id: str,
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    ok = await delete_chat(db, chat_id, user["sub"])
    if not ok:
        raise HTTPException(404, "Chat not found.")
    await delete_chat_session(chat_id)
    invalidate_retriever_cache(chat_id)
    return {"message": "Deleted."}


@router.post("/{chat_id}/message")
async def send_message(
    chat_id: str,
    req:     ChatMessageRequest,
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    return await process_chat(
        chat_id      = chat_id,
        user_message = req.message,
        llm_id       = req.llm_id,
        allow_search = req.allow_search,
        db           = db,
    )


@router.post("/{chat_id}/stream")
async def stream_message(
    chat_id: str,
    req:     ChatMessageRequest,
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")

    async def _events():
        try:
            async for token in stream_chat(
                chat_id      = chat_id,
                user_message = req.message,
                llm_id       = req.llm_id,
                allow_search = req.allow_search,
                db           = db,
            ):
                if token:
                    yield f"data: {token.replace(' ', chr(160))}\n\n"
        except Exception as e:
            logger.error(f"[STREAM] error: {e}")
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")