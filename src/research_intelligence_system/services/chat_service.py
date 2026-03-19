import asyncio
from typing import List

from src.research_intelligence_system.core.qa_system import run_qa_system
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ---------- CONFIG ----------
MAX_HISTORY_MESSAGES = 6   # tighter control for tokens
MAX_QUERY_LENGTH = 1500    # prevent prompt explosion


# ---------- CLEAN HISTORY ----------
def build_chat_context(messages: List) -> str:
    """
    Build structured + token-efficient chat history
    """

    if not messages:
        return ""

    # take last N messages only
    recent_msgs = messages[-MAX_HISTORY_MESSAGES:]

    formatted = []
    for m in recent_msgs:
        role = m.role.upper()
        content = m.content.strip()

        if content:
            formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


# ---------- QUERY OPTIMIZATION ----------
def build_enhanced_query(history: str, query: str) -> str:
    """
    Build safe + optimized query for RAG
    """

    query = query.strip()

    # truncate long queries
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]

    if not history:
        return query

    return f"""
Conversation Context:
{history}

User Question:
{query}

Rewrite the question if needed for better retrieval, but DO NOT change meaning.
"""


# ---------- MAIN ----------
async def process_chat(request):
    """
    🚀 Production Chat Service:
    - Async execution
    - Token control
    - Clean context building
    - Stable multi-turn handling
    """

    try:
        if not request.messages:
            return {
                "answer": "No input provided.",
                "source": "none",
                "confidence": 0.0
            }

        messages = request.messages
        latest_query = messages[-1].content.strip()

        # ---------- BUILD CONTEXT ----------
        history = build_chat_context(messages)

        enhanced_query = build_enhanced_query(history, latest_query)

        logger.info(f"[CHAT] chat_id={request.chat_id}")

        # ---------- CALL QA SYSTEM ----------
        response = await run_qa_system(
            query=enhanced_query,
            chat_id=request.chat_id,
            llm_id=request.llm_id,
            allow_search=request.allow_search
        )

        return response

    except Exception as e:
        logger.exception("Chat service failed.")

        return {
            "answer": "Something went wrong while processing your request.",
            "source": "error",
            "confidence": 0.0
        }