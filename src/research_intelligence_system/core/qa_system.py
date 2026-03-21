from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import AsyncIterator, Dict, Optional

from src.research_intelligence_system.constants import MIN_ANSWER, RAG_TIMEOUT, WEB_TIMEOUT
from src.research_intelligence_system.rag.llm import load_llm
from src.research_intelligence_system.rag.retriever import retrieve_documents_async
from src.research_intelligence_system.tools.web_search import async_web_search
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_TTL    = 300
_answer_cache: Dict[str, tuple] = {}


def _cache_key(query: str, chat_id: str) -> str:
    return hashlib.md5(f"{chat_id}:{query}".encode()).hexdigest()


# ── Formatting ────────────────────────────────────────────────────────────────
def _fix_formatting(text: str) -> str:
    # ensure newlines before numbered items
    text = re.sub(r'(\S)\s{0,2}(\d+\.\s+\*\*)', r'\1\n\n\2', text)
    # ensure newline before Summary
    text = re.sub(r'(\S)\s{0,2}(\*\*Summary)', r'\1\n\n\2', text)
    # collapse 3+ newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # fix missing spaces between words (SSE stripping artifact)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)
    text = re.sub(r'([,;:])([a-zA-Z])', r'\1 \2', text)
    return text.strip()


def _extract(response) -> str:
    if hasattr(response, "content"):
        text = response.content
    elif isinstance(response, dict):
        text = response.get("content", str(response))
    else:
        text = str(response)
    return text


def _is_good(text: str) -> bool:
    return bool(text) and "not found" not in text.lower() and len(text.strip()) >= MIN_ANSWER


def _synthesis_prompt(query: str, rag: str, web: str) -> str:
    return (
        "You are a research assistant. Answer using the sources below. "
        "Prefer paper content over web.\n\n"
        "FORMAT YOUR RESPONSE EXACTLY LIKE THIS EXAMPLE — DO NOT DEVIATE:\n\n"
        "1. **Concept one**: Explanation of the first concept.\n\n"
        "2. **Concept two**: Explanation of the second concept.\n\n"
        "3. **Concept three**: Explanation of the third concept.\n\n"
        "**Summary:** One sentence summary.\n\n"
        "RULES:\n"
        "- Every numbered item on its OWN LINE separated by a blank line\n"
        "- NO paragraph of text before the list\n"
        "- NO references or citations\n"
        "- Always put a SPACE between every word\n\n"
        f"Question: {query}\n\n"
        f"Paper Context:\n{rag}\n\n"
        f"Web Context:\n{web}\n"
    )


async def _safe(coro, timeout: float, label: str) -> str:
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result or ""
    except asyncio.TimeoutError:
        logger.warning(f"[QA] {label} timed out")
        return ""
    except Exception as e:
        logger.warning(f"[QA] {label} error: {e}")
        return ""


# ── Core ──────────────────────────────────────────────────────────────────────
async def run_qa_system(
    query: str,
    chat_id: str,
    llm_id: str,
    allow_search: bool = True,
) -> Dict:
    logger.info(f"[QA] chat_id={chat_id} query={query!r}")

    key   = _cache_key(query, chat_id)
    entry = _answer_cache.get(key)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        logger.info("[QA] cache hit")
        return {**entry[0], "cached": True}

    rag_coro = retrieve_documents_async(query, chat_id)
    web_coro = async_web_search(query) if allow_search else asyncio.sleep(0)

    rag_result, web_result = await asyncio.gather(
        _safe(rag_coro, RAG_TIMEOUT, "RAG"),
        _safe(web_coro, WEB_TIMEOUT, "web"),
    )

    if not rag_result and not web_result:
        return {"answer": "No relevant information found.", "source": "none", "confidence": 0.2}

    llm    = load_llm(llm_id)
    prompt = _synthesis_prompt(query, rag_result, web_result)

    try:
        raw    = await asyncio.to_thread(llm.invoke, prompt)
        answer = _fix_formatting(_extract(raw))
    except Exception as e:
        logger.error(f"[QA] LLM failed: {e}")
        answer = rag_result or web_result or "Failed to generate answer."

    source = "hybrid" if (rag_result and web_result) else ("rag" if rag_result else "web")
    result = {"answer": answer, "source": source, "confidence": 0.85}
    _answer_cache[key] = (result, time.time())
    return result


# ── Streaming — yields full formatted answer as single chunk ──────────────────
async def stream_qa(
    query: str,
    chat_id: str,
    llm_id: str,
    allow_search: bool = True,
) -> AsyncIterator[str]:
    """
    Collects full LLM response then yields it as ONE chunk.
    SSE token-by-token streaming strips spaces — single chunk avoids this.
    """
    rag_coro = retrieve_documents_async(query, chat_id)
    web_coro = async_web_search(query) if allow_search else asyncio.sleep(0)

    rag_result, web_result = await asyncio.gather(
        _safe(rag_coro, RAG_TIMEOUT, "RAG"),
        _safe(web_coro, WEB_TIMEOUT, "web"),
    )

    if not rag_result and not web_result:
        yield "No relevant information found."
        return

    llm    = load_llm(llm_id)
    prompt = _synthesis_prompt(query, rag_result, web_result)

    try:
        if hasattr(llm, "astream"):
            full = ""
            async for chunk in llm.astream(prompt):
                full += _extract(chunk)
            yield _fix_formatting(full)
        else:
            raw = await asyncio.to_thread(llm.invoke, prompt)
            yield _fix_formatting(_extract(raw))
    except Exception as e:
        logger.error(f"[QA] stream failed: {e}")
        yield rag_result or "Failed to generate answer."


# ── Cache management ──────────────────────────────────────────────────────────
def invalidate_qa_cache(chat_id: Optional[str] = None) -> None:
    if chat_id:
        drop = [k for k in _answer_cache if chat_id in k]
        for k in drop: del _answer_cache[k]
    else:
        _answer_cache.clear()