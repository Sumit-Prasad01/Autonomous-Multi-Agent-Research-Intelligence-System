from __future__ import annotations

import re
import asyncio
import hashlib
from typing import AsyncIterator, Dict, Optional

from src.research_intelligence_system.rag.retriever import retrieve_documents_async
from src.research_intelligence_system.rag.llm import load_llm
from src.research_intelligence_system.tools.web_search import run_web_search
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RAG_TIMEOUT = 8
WEB_TIMEOUT = 10
MIN_ANSWER  = 30

# ── Answer cache (query+chat_id → result) ────────────────────────────────────
_answer_cache: Dict[str, Dict] = {}

def _cache_key(query: str, chat_id: str) -> str:
    return hashlib.md5(f"{chat_id}:{query}".encode()).hexdigest()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_good(text: str) -> bool:
    return bool(text) and "not found" not in text.lower() and len(text.strip()) >= MIN_ANSWER

def _fix_formatting(text: str) -> str:
    """Force blank lines between numbered items — fixes small LLMs ignoring the instruction."""
    text = re.sub(r"(\S)\s*(\d+\.\s+\*\*)", r"\g<1>\g<2>", text)
    text = re.sub(r"(\S)\s*(\*\*Summary)", r"\g<1>\g<2>", text)
    return text.strip()

def _extract(response) -> str:
    if hasattr(response, "content"): text = response.content
    elif isinstance(response, dict): text = response.get("content", str(response))
    else: text = str(response)
    return _fix_formatting(text)

def _synthesis_prompt(query: str, rag: str, web: str) -> str:
    return f"""You are a research assistant. Answer using the sources below. Prefer paper content over web.

            STRICT formatting rules:
            - Start directly with the list. NO bold title or heading before the list.
            - Each numbered item MUST be on its own line with a blank line after it.
            - Bold only the concept name: **Concept**: explanation here.
            - Do NOT run items together on one line.
            - Do NOT add references or citations.
            - End with a **Summary:** line.

            Correct example output:
            1. **Fine-tuning**: Adds a classification layer and trains all parameters end-to-end.

            2. **Feature-based**: Extracts fixed embeddings from pre-trained layers without fine-tuning.

            **Summary:** Both approaches leverage pre-trained BERT representations for downstream tasks.

            ---
            Question: {query}

            Paper Context:
            {rag}

            Web Context:
            {web}
    """

async def _safe(coro, timeout: float, label: str) -> str:
    try:
        return await asyncio.wait_for(coro, timeout=timeout) or ""
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

    # ── cache hit ─────────────────────────────────────────────────────────────
    key = _cache_key(query, chat_id)
    if key in _answer_cache:
        logger.info("[QA] cache hit")
        return {**_answer_cache[key], "cached": True}

    # ── parallel RAG + web (if allowed) ──────────────────────────────────────
    rag_coro = retrieve_documents_async(query, chat_id)
    web_coro = asyncio.to_thread(run_web_search, query) if allow_search else asyncio.sleep(0)

    rag_result, web_result = await asyncio.gather(
        _safe(rag_coro, RAG_TIMEOUT, "RAG"),
        _safe(web_coro, WEB_TIMEOUT, "web"),
    )

    # ── no context at all ────────────────────────────────────────────────────
    if not rag_result and not web_result:
        return {"answer": "No relevant information found.", "source": "none", "confidence": 0.2}

    # ── always synthesise through LLM for consistent markdown formatting ─────
    llm    = load_llm(llm_id)
    prompt = _synthesis_prompt(query, rag_result, web_result)

    try:
        raw    = await asyncio.to_thread(llm.invoke, prompt)
        answer = _extract(raw).strip()
    except Exception as e:
        logger.error(f"[QA] LLM synthesis failed: {e}")
        answer = rag_result or web_result or "Failed to generate answer."

    source = "hybrid" if (rag_result and web_result) else ("rag" if rag_result else "web")
    result = {"answer": answer, "source": source, "confidence": 0.85}
    _answer_cache[key] = result
    return result


# ── Streaming variant ─────────────────────────────────────────────────────────
async def stream_qa(
    query: str,
    chat_id: str,
    llm_id: str,
    allow_search: bool = True,
) -> AsyncIterator[str]:
    """
    Yields answer tokens as they arrive.
    Falls back to run_qa_system if LLM doesn't support streaming.
    """
    rag_coro = retrieve_documents_async(query, chat_id)
    web_coro = asyncio.to_thread(run_web_search, query) if allow_search else asyncio.sleep(0)

    rag_result, web_result = await asyncio.gather(
        _safe(rag_coro, RAG_TIMEOUT, "RAG"),
        _safe(web_coro, WEB_TIMEOUT, "web"),
    )

    llm    = load_llm(llm_id)
    prompt = _synthesis_prompt(query, rag_result, web_result)

    if hasattr(llm, "astream"):
        async for chunk in llm.astream(prompt):
            yield _extract(chunk)
    else:
        result = await asyncio.to_thread(llm.invoke, prompt)
        yield _extract(result)


# ── Cache management ──────────────────────────────────────────────────────────
def invalidate_qa_cache(chat_id: Optional[str] = None) -> None:
    if chat_id:
        drop = [k for k in _answer_cache if chat_id in k]
        for k in drop: del _answer_cache[k]
    else:
        _answer_cache.clear()
