from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.rag.reranker import rerank_documents
from src.research_intelligence_system.rag.vector_store import (
    _store,
    async_search_with_score,
    load_vector_store,
)
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DENSE_FETCH_K  = 20          # over-fetch before rerank
BM25_FETCH_K   = 20
FINAL_TOP_K    = 5           # returned to QA system
RRF_K          = 60          # RRF constant (higher = smoother rank blending)
_POOL          = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retriever")

# Section keywords → route to relevant section metadata
_SECTION_HINTS: Dict[str, str] = {
    "method":      "methodology",
    "approach":    "methodology",
    "result":      "results",
    "finding":     "results",
    "conclusion":  "conclusion",
    "abstract":    "abstract",
    "discuss":     "discussion",
}


# ── Per-chat BM25 cache ───────────────────────────────────────────────────────
class _BM25Cache:
    """
    Stores one BM25Retriever per chat_id.
    Invalidated when new docs are ingested for that chat.
    """
    def __init__(self):
        self._cache: Dict[str, Tuple[BM25Retriever, float]] = {}   # chat_id → (retriever, ts)

    def get(self, chat_id: str) -> Optional[BM25Retriever]:
        entry = self._cache.get(chat_id)
        return entry[0] if entry else None

    def set(self, chat_id: str, retriever: BM25Retriever):
        self._cache[chat_id] = (retriever, time.time())

    def invalidate(self, chat_id: str):
        self._cache.pop(chat_id, None)
        logger.debug(f"[BM25 CACHE] invalidated chat_id={chat_id}")

    def invalidate_all(self):
        self._cache.clear()


_bm25_cache = _BM25Cache()


def _build_bm25(chat_id: str) -> BM25Retriever:
    """Build BM25 index from docs belonging to this chat only."""
    db   = load_vector_store()
    docs = [
        doc for doc in db.docstore._dict.values()
        if doc.metadata.get("chat_id") == chat_id
    ]
    if not docs:
        raise CustomException(f"No documents found for chat_id={chat_id}")

    bm25   = BM25Retriever.from_documents(docs)
    bm25.k = BM25_FETCH_K
    logger.info(f"[BM25] built index: {len(docs)} docs [chat_id={chat_id}]")
    return bm25


async def _get_bm25(chat_id: str) -> BM25Retriever:
    cached = _bm25_cache.get(chat_id)
    if cached:
        return cached
    loop    = asyncio.get_event_loop()
    bm25    = await loop.run_in_executor(_POOL, _build_bm25, chat_id)
    _bm25_cache.set(chat_id, bm25)
    return bm25


# ── RRF fusion ────────────────────────────────────────────────────────────────
def _rrf_fuse(
    dense_results: List[Tuple[Document, float]],
    bm25_results:  List[Document],
) -> List[Document]:
    """
    Reciprocal Rank Fusion — normalises across score distributions
    so BM25 ordinal ranks and FAISS L2 scores are comparable.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results):
        key = doc.page_content[:120]
        scores[key]  = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:120]
        scores[key]  = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in ranked]


# ── Section hint detection ─────────────────────────────────────────────────────
def _infer_section(query: str) -> Optional[str]:
    q = query.lower()
    for kw, section in _SECTION_HINTS.items():
        if kw in q:
            return section
    return None


# ── Core async retrieval ──────────────────────────────────────────────────────
async def _hybrid_fetch(
    query: str,
    chat_id: str,
    section_filter: Optional[str],
) -> List[Document]:
    """Run dense + BM25 retrieval in parallel, fuse with RRF."""

    # build filter
    dense_filter = {"chat_id": chat_id}
    if section_filter:
        dense_filter["section"] = section_filter

    # parallel fetch
    dense_task = async_search_with_score(query, k=DENSE_FETCH_K, chat_id=chat_id)
    bm25_obj   = await _get_bm25(chat_id)

    loop       = asyncio.get_event_loop()
    bm25_task  = loop.run_in_executor(_POOL, bm25_obj.invoke, query)

    dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)

    logger.info(
        f"[RETRIEVER] dense={len(dense_results)} bm25={len(bm25_results)} "
        f"section={section_filter}"
    )

    return _rrf_fuse(dense_results, bm25_results)


# ── Public async API ──────────────────────────────────────────────────────────
async def retrieve_documents_async(
    query: str,
    chat_id: str,
    top_k: int = FINAL_TOP_K,
) -> str:
    """
    Full async retrieval pipeline:
      dense + BM25 (parallel) → RRF fusion → reranker → format
    """
    logger.info(f"[RETRIEVER] query={query!r} chat_id={chat_id}")

    section = _infer_section(query)
    fused   = await _hybrid_fetch(query, chat_id, section)

    if not fused:
        logger.warning("[RETRIEVER] no documents found")
        return ""

    # ── rerank fused candidates ───────────────────────────────────────────────
    texts  = [d.page_content for d in fused]
    loop   = asyncio.get_event_loop()
    ranked = await loop.run_in_executor(
        _POOL, rerank_documents, query, texts, top_k
    )

    logger.info(f"[RETRIEVER] returning {len(ranked)} chunks after rerank")
    return "\n\n".join(ranked)


# ── Sync shim (backward compat) ───────────────────────────────────────────────
def retrieve_documents(query: str, chat_id: str, llm_id: str = "") -> str:
    """
    Sync wrapper for legacy callers.
    llm_id retained for signature compatibility but no longer used
    (LLMChainExtractor removed — saves N Groq calls per query).
    """
    try:
        return asyncio.get_event_loop().run_until_complete(
            retrieve_documents_async(query, chat_id)
        )
    except Exception as e:
        logger.exception("retrieve_documents failed")
        raise CustomException("Retrieval failed", e)


# ── Cache management (call after ingesting new docs) ─────────────────────────
def invalidate_retriever_cache(chat_id: Optional[str] = None) -> None:
    if chat_id:
        _bm25_cache.invalidate(chat_id)
    else:
        _bm25_cache.invalidate_all()
