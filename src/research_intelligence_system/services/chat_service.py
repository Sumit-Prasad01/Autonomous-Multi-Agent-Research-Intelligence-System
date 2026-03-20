"""
chat_service.py — Async, streaming, session-cached chat service
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncIterator, Dict, List, Optional

from src.research_intelligence_system.core.qa_system import run_qa_system, stream_qa
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_HISTORY  = 6
MAX_Q_CHARS  = 1500

# ── Session history cache (chat_id → last built context str) ─────────────────
_session_cache: Dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_history(messages: List) -> str:
    return "\n".join(
        f"{m.role.upper()}: {m.content.strip()}"
        for m in messages[-MAX_HISTORY:]
        if m.content.strip()
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


def _session_key(chat_id: str, query: str) -> str:
    return hashlib.md5(f"{chat_id}:{query}".encode()).hexdigest()


# ── Main ──────────────────────────────────────────────────────────────────────
async def process_chat(request) -> Dict:
    if not request.messages:
        return {"answer": "No input provided.", "source": "none", "confidence": 0.0}

    latest   = request.messages[-1].content.strip()
    history  = _build_history(request.messages[:-1])
    query    = _build_query(history, latest)

    # cache context for streaming reuse
    _session_cache[request.chat_id] = history

    logger.info(f"[CHAT] chat_id={request.chat_id} query={latest!r}")

    try:
        return await run_qa_system(
            query=query,
            chat_id=request.chat_id,
            llm_id=request.llm_id,
            allow_search=request.allow_search,
        )
    except Exception as e:
        logger.exception("[CHAT] failed")
        return {"answer": "Something went wrong.", "source": "error", "confidence": 0.0}


async def stream_chat(request) -> AsyncIterator[str]:
    """Streaming variant — yields tokens for SSE / WebSocket delivery."""
    if not request.messages:
        yield "No input provided."
        return

    latest  = request.messages[-1].content.strip()
    history = _session_cache.get(request.chat_id, _build_history(request.messages[:-1]))
    query   = _build_query(history, latest)

    logger.info(f"[CHAT STREAM] chat_id={request.chat_id}")

    try:
        async for token in stream_qa(
            query=query,
            chat_id=request.chat_id,
            llm_id=request.llm_id,
            allow_search=request.allow_search,
        ):
            yield token
    except Exception as e:
        logger.exception("[CHAT STREAM] failed")
        yield "Something went wrong."


def clear_session(chat_id: Optional[str] = None) -> None:
    if chat_id:
        _session_cache.pop(chat_id, None)
    else:
        _session_cache.clear()
