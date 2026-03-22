from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.research_intelligence_system.constants import (
    BM25_FETCH_K, COLLECTION_NAME, DENSE_FETCH_K, FINAL_TOP_K, RRF_K
)
from src.research_intelligence_system.rag.reranker import rerank_documents
from src.research_intelligence_system.rag.vector_store import (
    _store, async_search_with_score
)
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retriever")

_SECTION_HINTS: Dict[str, str] = {
    "method":     "methodology",
    "approach":   "methodology",
    "result":     "results",
    "finding":    "results",
    "conclusion": "conclusion",
    "abstract":   "abstract",
    "discuss":    "discussion",
}


# ── Per-chat BM25 cache ───────────────────────────────────────────────────────
class _BM25Cache:
    def __init__(self):
        self._cache: Dict[str, Tuple[BM25Retriever, float]] = {}

    def get(self, chat_id: str) -> Optional[BM25Retriever]:
        entry = self._cache.get(chat_id)
        return entry[0] if entry else None

    def set(self, chat_id: str, r: BM25Retriever):
        self._cache[chat_id] = (r, time.time())

    def invalidate(self, chat_id: str):
        self._cache.pop(chat_id, None)

    def invalidate_all(self):
        self._cache.clear()


_bm25_cache = _BM25Cache()


def _build_bm25(chat_id: str) -> BM25Retriever:
    """Fetch docs from Qdrant scroll API — works with QdrantVectorStore."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    results, _ = _store.client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(
            key="metadata.chat_id",
            match=MatchValue(value=chat_id),
        )]),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    if results:
        logger.info(f"[BM25 DEBUG] first point payload: {results[0].payload}")
    else:
        logger.warning(f"[BM25 DEBUG] no points found for chat_id={chat_id}")

    docs = []
    for point in results:
        payload = point.payload or {}
        # LangChain QdrantVectorStore stores content under "page_content" key
        content  = payload.get("page_content", "")
        metadata = payload.get("metadata", {})
        if content:
            docs.append(Document(page_content=content, metadata=metadata))

    if not docs:
        raise CustomException(f"No documents for chat_id={chat_id}")

    bm25   = BM25Retriever.from_documents(docs)
    bm25.k = BM25_FETCH_K
    logger.info(f"[BM25] {len(docs)} docs [chat_id={chat_id}]")
    return bm25


async def _get_bm25(chat_id: str) -> BM25Retriever:
    cached = _bm25_cache.get(chat_id)
    if cached:
        return cached
    loop = asyncio.get_running_loop()
    bm25 = await loop.run_in_executor(_POOL, _build_bm25, chat_id)
    _bm25_cache.set(chat_id, bm25)
    return bm25


# ── RRF fusion ────────────────────────────────────────────────────────────────
def _rrf_fuse(
    dense: List[Tuple[Document, float]],
    bm25:  List[Document],
) -> List[Document]:
    scores:  Dict[str, float]    = {}
    doc_map: Dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense):
        key = doc.page_content[:120]
        scores[key]  = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25):
        key = doc.page_content[:120]
        scores[key]  = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    return [doc_map[k] for k in sorted(scores, key=lambda k: scores[k], reverse=True)]


def _infer_section(query: str) -> Optional[str]:
    q = query.lower()
    for kw, section in _SECTION_HINTS.items():
        if kw in q:
            return section
    return None


# ── Core ──────────────────────────────────────────────────────────────────────
async def _hybrid_fetch(query: str, chat_id: str,
                        section: Optional[str]) -> List[Document]:
    loop = asyncio.get_running_loop()

    # run dense + BM25 in parallel
    try:
        bm25_obj = await _get_bm25(chat_id)
        dense_task = async_search_with_score(query, k=DENSE_FETCH_K, chat_id=chat_id)
        bm25_task  = loop.run_in_executor(_POOL, bm25_obj.invoke, query)
        dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
    except CustomException:
        # BM25 failed (no docs in Qdrant yet) — fall back to dense only
        logger.warning("[RETRIEVER] BM25 failed — falling back to dense only")
        dense_results = await async_search_with_score(query, k=DENSE_FETCH_K, chat_id=chat_id)
        bm25_results  = []

    logger.info(f"[RETRIEVER] dense={len(dense_results)} bm25={len(bm25_results)} section={section}")
    return _rrf_fuse(dense_results, bm25_results)


async def retrieve_documents_async(
    query: str,
    chat_id: str,
    top_k: int = FINAL_TOP_K,
) -> str:
    logger.info(f"[RETRIEVER] query={query!r} chat_id={chat_id}")

    section = _infer_section(query)
    fused   = await _hybrid_fetch(query, chat_id, section)

    if not fused:
        logger.warning("[RETRIEVER] no documents found")
        return ""

    loop   = asyncio.get_running_loop()
    texts  = [d.page_content for d in fused]
    ranked = await loop.run_in_executor(_POOL, rerank_documents, query, texts, top_k)

    logger.info(f"[RETRIEVER] {len(ranked)} chunks after rerank")
    return "\n\n".join(ranked)


# ── Sync shim ─────────────────────────────────────────────────────────────────
def retrieve_documents(query: str, chat_id: str, llm_id: str = "") -> str:
    try:
        return asyncio.get_event_loop().run_until_complete(
            retrieve_documents_async(query, chat_id)
        )
    except Exception as e:
        raise CustomException("Retrieval failed", e)


# ── Cache management ──────────────────────────────────────────────────────────
def invalidate_retriever_cache(chat_id: Optional[str] = None) -> None:
    if chat_id:
        _bm25_cache.invalidate(chat_id)
    else:
        _bm25_cache.invalidate_all()