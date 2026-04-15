from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import AsyncIterator, Dict, List, Optional

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
    text = re.sub(r'(\S)\s{0,2}(\d+\.\s+\*\*)', r'\1\n\n\2', text)
    text = re.sub(r'(\S)\s{0,2}(\*\*Summary)', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)
    text = re.sub(r'([,;:])([a-zA-Z])', r'\1 \2', text)
    return text.strip()


def _extract(response) -> str:
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict):
        return response.get("content", str(response))
    return str(response)


def _extract_question(query: str) -> str:
    if "User Question:" in query:
        query = query.split("User Question:")[-1].strip()
    if "\n\nAnswer using" in query:
        query = query.split("\n\nAnswer using")[0].strip()
    return query


def _compute_confidence(
    query:         str,
    rag_result:    str,
    web_result:    str,
    paper_context: str,
) -> float:
    """
    Compute answer confidence based on:
    - RAG coverage: keyword overlap between query and retrieved chunks
    - Source richness: how many sources contributed
    - Content length: longer RAG result = more relevant chunks found
    """
    if not rag_result and not web_result and not paper_context:
        return 0.1

    score = 0.0

    # source richness (0.0 - 0.3)
    sources = sum([bool(rag_result), bool(web_result), bool(paper_context)])
    score  += sources * 0.1

    # RAG keyword overlap (0.0 - 0.4)
    if rag_result:
        query_words  = set(query.lower().split())
        result_words = set(rag_result.lower().split())
        if query_words:
            overlap = len(query_words & result_words) / len(query_words)
            score  += min(overlap * 0.4, 0.4)

    # RAG content length (0.0 - 0.2)
    if rag_result:
        length_score = min(len(rag_result) / 1000, 1.0) * 0.2
        score       += length_score

    # paper context bonus (0.0 - 0.1)
    if paper_context and len(paper_context) > 100:
        score += 0.1

    return round(min(score, 1.0), 2)


def _is_good(text: str) -> bool:
    return bool(text) and "not found" not in text.lower() and len(text.strip()) >= MIN_ANSWER


# ── Paper analysis context ────────────────────────────────────────────────────
async def _get_paper_context(chat_id: str) -> str:
    """
    Fetch structured paper analysis from Postgres and inject into prompt.
    This gives LLM access to entities, refined summary, and research gaps.
    """
    try:
        from src.research_intelligence_system.database.database import AsyncSessionLocal
        from src.research_intelligence_system.database.paper_repository import get_paper_analyses

        async with AsyncSessionLocal() as db:
            analyses = await get_paper_analyses(db, chat_id)

        if not analyses:
            return ""

        parts = []
        for a in analyses:
            entities = a.entities or {}
            part     = f"Paper: {a.filename or 'Unknown'}\n"

            if a.refined_summary:
                part += f"Summary: {a.refined_summary[:400]}\n"

            if entities.get("models"):
                part += f"Models: {', '.join(entities['models'][:8])}\n"
            if entities.get("datasets"):
                part += f"Datasets: {', '.join(entities['datasets'][:8])}\n"
            if entities.get("metrics"):
                part += f"Metrics: {', '.join(entities['metrics'][:8])}\n"
            if entities.get("methods"):
                part += f"Methods: {', '.join(entities['methods'][:8])}\n"

            if a.research_gaps:
                part += f"Research Gaps: {'; '.join(a.research_gaps[:3])}\n"

            parts.append(part)

        return "\n---\n".join(parts)

    except Exception as e:
        logger.debug(f"[QA] paper context fetch failed (non-fatal): {e}")
        return ""


# ── Prompts ───────────────────────────────────────────────────────────────────
def _synthesis_prompt(
    query:         str,
    rag:           str,
    web:           str,
    paper_context: str = "",
) -> str:
    paper_section = (
        f"Structured Paper Analysis:\n{paper_context}\n\n"
        if paper_context else ""
    )
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
        f"{paper_section}"
        f"Paper Context (retrieved chunks):\n{rag}\n\n"
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


# ── Core QA ───────────────────────────────────────────────────────────────────
async def run_qa_system(
    query:        str,
    chat_id:      str,
    llm_id:       str,
    allow_search: bool = True,
) -> Dict:
    logger.info(f"[QA] chat_id={chat_id} query={query!r}")

    # cache check
    key   = _cache_key(query, chat_id)
    entry = _answer_cache.get(key)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        logger.info("[QA] cache hit")
        cached        = entry[0].copy()
        cached["answer"] = _fix_formatting(cached["answer"])
        return {**cached, "cached": True}

    clean_query = _extract_question(query)

    # parallel: GraphRAG + web search + paper analysis context
    rag_coro     = retrieve_documents_async(clean_query, chat_id)
    web_coro     = async_web_search(clean_query) if allow_search else asyncio.sleep(0)
    context_coro = _get_paper_context(chat_id)

    rag_result, web_result, paper_context = await asyncio.gather(
        _safe(rag_coro,     RAG_TIMEOUT, "RAG"),
        _safe(web_coro,     WEB_TIMEOUT, "web"),
        _safe(context_coro, 5.0,         "paper_context"),
    )

    if not rag_result and not web_result and not paper_context:
        return {"answer": "No relevant information found.", "source": "none", "confidence": 0.2}

    llm    = load_llm(llm_id)
    prompt = _synthesis_prompt(query, rag_result, web_result, paper_context)

    try:
        raw    = await asyncio.to_thread(llm.invoke, prompt)
        answer = _fix_formatting(_extract(raw))
    except Exception as e:
        logger.error(f"[QA] LLM failed: {e}")
        answer = rag_result or paper_context or "Failed to generate answer."

    source     = "hybrid" if (rag_result and web_result) else ("rag" if rag_result else "web")
    confidence = _compute_confidence(clean_query, rag_result, web_result, paper_context)
    result     = {"answer": answer, "source": source, "confidence": confidence}

    if len(answer) > 100:
        _answer_cache[key] = (result, time.time())

    return result


# ── Streaming ─────────────────────────────────────────────────────────────────
async def stream_qa(
    query:        str,
    chat_id:      str,
    llm_id:       str,
    allow_search: bool = True,
) -> AsyncIterator[str]:
    clean_query = _extract_question(query)

    rag_coro     = retrieve_documents_async(clean_query, chat_id)
    web_coro     = async_web_search(clean_query) if allow_search else asyncio.sleep(0)
    context_coro = _get_paper_context(chat_id)

    rag_result, web_result, paper_context = await asyncio.gather(
        _safe(rag_coro,     RAG_TIMEOUT, "RAG"),
        _safe(web_coro,     WEB_TIMEOUT, "web"),
        _safe(context_coro, 5.0,         "paper_context"),
    )

    if not rag_result and not web_result and not paper_context:
        yield "No relevant information found."
        return

    llm    = load_llm(llm_id)
    prompt = _synthesis_prompt(query, rag_result, web_result, paper_context)

    try:
        if hasattr(llm, "astream"):
            full = ""
            async for chunk in llm.astream(prompt):
                if hasattr(chunk, "content") and chunk.content:
                    full += chunk.content
                    yield chunk.content
        else:
            result = await asyncio.to_thread(llm.invoke, prompt)
            yield _extract(result)
    except Exception as e:
        logger.error(f"[QA] stream failed: {e}")
        yield rag_result or paper_context or "Failed to generate answer."


# ── Cache management ──────────────────────────────────────────────────────────
def invalidate_qa_cache(chat_id: Optional[str] = None) -> None:
    if chat_id:
        drop = [k for k in _answer_cache if chat_id in k]
        for k in drop:
            del _answer_cache[k]
    else:
        _answer_cache.clear()