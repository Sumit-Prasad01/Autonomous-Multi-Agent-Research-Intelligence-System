from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.research_intelligence_system.database.models import Chat, Message
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ── Chat ──────────────────────────────────────────────────────────────────────
async def create_chat(
    db: AsyncSession,
    user_id: str,
    title: str = "New Chat",
    llm_id: str = "llama-3.1-8b-instant",
    allow_search: bool = False,
) -> Chat:
    chat = Chat(
        user_id=uuid.UUID(user_id),
        title=title,
        llm_id=llm_id,
        allow_search=allow_search,
    )
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    return chat


async def get_user_chats(db: AsyncSession, user_id: str) -> List[Chat]:
    result = await db.execute(
        select(Chat)
        .where(Chat.user_id == uuid.UUID(user_id), Chat.is_deleted == False)
        .order_by(Chat.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_chat(
    db: AsyncSession, chat_id: str, user_id: str
) -> Optional[Chat]:
    return await db.scalar(
        select(Chat)
        .where(
            Chat.id == uuid.UUID(chat_id),
            Chat.user_id == uuid.UUID(user_id),
            Chat.is_deleted == False,
        )
        .options(selectinload(Chat.messages))
    )


async def update_chat_title(db: AsyncSession, chat_id: str, title: str) -> None:
    await db.execute(
        update(Chat)
        .where(Chat.id == uuid.UUID(chat_id))
        .values(title=title)
    )
    await db.commit()


async def delete_chat(db: AsyncSession, chat_id: str, user_id: str) -> bool:
    """Soft delete — sets is_deleted=True instead of removing the row."""
    result = await db.execute(
        update(Chat)
        .where(
            Chat.id == uuid.UUID(chat_id),
            Chat.user_id == uuid.UUID(user_id),
        )
        .values(is_deleted=True)
    )
    await db.commit()
    return result.rowcount > 0


# ── Message ───────────────────────────────────────────────────────────────────
async def save_message(
    db: AsyncSession,
    chat_id: str,
    role: str,
    content: str,
    source: str = "",
    confidence: str = "",
) -> Message:
    msg = Message(
        chat_id=uuid.UUID(chat_id),
        role=role,
        content=content,
        source=source,
        confidence=confidence,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_chat_messages(db: AsyncSession, chat_id: str) -> List[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.chat_id == uuid.UUID(chat_id))
        .order_by(Message.created_at)
    )
    return list(result.scalars().all())