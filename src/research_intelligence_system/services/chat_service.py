from __future__ import annotations

import asyncio
from typing import AsyncIterator, Dict, List

from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.constants import MAX_HISTORY, MAX_Q_CHARS
from src.research_intelligence_system.core.qa_system import (
    invalidate_qa_cache, run_qa_system, stream_qa
)
from src.research_intelligence_system.database.chat_repository import (
    get_chat_messages, save_message, update_chat_title
)
from src.research_intelligence_system.services.redis_service import (
    clear_memory, get_memory, push_message
)
from src.research_intelligence_system.utils.logger import get_logger

from src.research_intelligence_system.core.qa_system import _fix_formatting

logger = get_logger(__name__)


def _build_history(messages: List[dict]) -> str:
    return "\n".join(
        f"{m['role'].upper()}: {m['content'].strip()}"
        for m in messages[-MAX_HISTORY:]
        if m["content"].strip()
    )


def _build_query(history: str, query: str) -> str:
    query = query.strip()[:MAX_Q_CHARS]
    if not history:
        return query
    return (
        f"Conversation Context:\n{history}\n\n"
        f"User Question:\n{query}\n\n"
        "Answer using the context above if relevant."
    )


async def process_chat(
    chat_id: str,
    user_message: str,
    llm_id: str,
    allow_search: bool,
    db: AsyncSession,
) -> Dict:
    memory  = await get_memory(chat_id)
    history = _build_history(memory)
    query   = _build_query(history, user_message)

    logger.info(f"[CHAT] chat_id={chat_id} query={user_message!r}")

    # update title BEFORE pushing to memory so not memory check is accurate
    if not memory:
        await update_chat_title(db, chat_id, user_message[:60])

    await asyncio.gather(
        push_message(chat_id, "user", user_message),
        save_message(db, chat_id, "user", user_message),
    )

    try:
        result = await run_qa_system(
            query=query,
            chat_id=chat_id,
            llm_id=llm_id,
            allow_search=allow_search,
        )
    except Exception:
        logger.exception("[CHAT] QA failed")
        result = {"answer": "Something went wrong.", "source": "error", "confidence": 0.0}

    answer = result.get("answer", "")

    # save assistant reply to Redis + Postgres concurrently
    await asyncio.gather(
        push_message(chat_id, "assistant", answer),
        save_message(
            db, chat_id, "assistant", answer,
            source=result.get("source", ""),
            confidence=str(result.get("confidence", "")),
        ),
    )
    return result


async def stream_chat(
    chat_id: str,
    user_message: str,
    llm_id: str,
    allow_search: bool,
    db: AsyncSession,
) -> AsyncIterator[str]:
    memory  = await get_memory(chat_id)
    history = _build_history(memory)
    query   = _build_query(history, user_message)

    # update title on first message
    if not memory:
        await update_chat_title(db, chat_id, user_message[:60])

    await asyncio.gather(
        push_message(chat_id, "user", user_message),
        save_message(db, chat_id, "user", user_message),
    )

    full = ""
    try:
        async for token in stream_qa(
            query=query, chat_id=chat_id,
            llm_id=llm_id, allow_search=allow_search,
        ):
            full += token
            yield token
    except Exception:
        logger.exception("[CHAT STREAM] failed")
        yield "Something went wrong."

    if full:
        full = _fix_formatting(full)
        await asyncio.gather(
            push_message(chat_id, "assistant", full),
            save_message(db, chat_id, "assistant", full, source="stream"),
        )

        
async def load_chat_history(chat_id: str, db: AsyncSession) -> List[dict]:
    msgs = await get_chat_messages(db, chat_id)
    return [
        {"role": m.role, "content": m.content,
         "source": m.source, "confidence": m.confidence}
        for m in msgs
    ]


async def delete_chat_session(chat_id: str) -> None:
    """Clear Redis memory + QA cache for a chat."""
    await asyncio.gather(
        clear_memory(chat_id),
        asyncio.to_thread(invalidate_qa_cache, chat_id),
    )