"""
retriever.py — Hybrid GraphRAG retriever
Combines: Dense (Qdrant) + BM25 + RRF fusion + Reranker + Neo4j subgraph
"""
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


# ── BM25 cache ────────────────────────────────────────────────────────────────
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

    docs = []
    for point in results:
        payload = point.payload or {}
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


# ── GraphRAG: Neo4j subgraph context ─────────────────────────────────────────
def _extract_query_entities(query: str) -> List[str]:
    """
    Extract potential entity names from query for Neo4j lookup.
    Simple heuristic: capitalized words + known keywords.
    """
    words   = query.split()
    # capitalized words likely to be entity names
    entities = [w.strip("?.,!:;") for w in words if w and w[0].isupper()]
    # also add 2-word combos (e.g. "BERT model", "ImageNet dataset")
    for i in range(len(words) - 1):
        combo = f"{words[i]} {words[i+1]}"
        if words[i][0].isupper():
            entities.append(combo.strip("?.,!:;"))
    return list(set(entities))[:10]


def _get_graph_context(chat_id: str, query: str) -> str:
    """
    Query Neo4j for subgraph around query entities.
    Returns natural language representation of graph context.
    """
    try:
        from src.research_intelligence_system.knowledge_graph.neo4j_service import Neo4jService
        neo4j    = Neo4jService()
        entities = _extract_query_entities(query)
        if not entities:
            return ""
        context = neo4j.get_subgraph_text(chat_id, entities)
        if context:
            logger.info(f"[GRAPHRAG] got {len(context.splitlines())} triples from Neo4j")
        return context
    except Exception as e:
        logger.debug(f"[GRAPHRAG] Neo4j context failed (non-fatal): {e}")
        return ""


# ── Core hybrid fetch ─────────────────────────────────────────────────────────
async def _hybrid_fetch(
    query:   str,
    chat_id: str,
    section: Optional[str],
) -> Tuple[List[Document], str]:
    """
    Run dense + BM25 + Neo4j in parallel.
    Returns (fused_docs, graph_context).
    """
    loop = asyncio.get_running_loop()

    # dense + BM25 + graph — all three in parallel
    try:
        bm25_obj = await _get_bm25(chat_id)

        dense_task = async_search_with_score(query, k=DENSE_FETCH_K, chat_id=chat_id)
        bm25_task  = loop.run_in_executor(_POOL, bm25_obj.invoke, query)
        graph_task = loop.run_in_executor(_POOL, _get_graph_context, chat_id, query)

        dense_results, bm25_results, graph_context = await asyncio.gather(
            dense_task, bm25_task, graph_task
        )

    except CustomException:
        logger.warning("[RETRIEVER] BM25 failed — falling back to dense + graph")
        dense_task = async_search_with_score(query, k=DENSE_FETCH_K, chat_id=chat_id)
        graph_task = loop.run_in_executor(_POOL, _get_graph_context, chat_id, query)
        dense_results, graph_context = await asyncio.gather(dense_task, graph_task)
        bm25_results = []

    logger.info(
        f"[RETRIEVER] dense={len(dense_results)} "
        f"bm25={len(bm25_results)} "
        f"graph_triples={len(graph_context.splitlines()) if graph_context else 0} "
        f"section={section}"
    )

    fused = _rrf_fuse(dense_results, bm25_results)
    return fused, graph_context or ""


# ── Public async API ──────────────────────────────────────────────────────────
async def retrieve_documents_async(
    query:   str,
    chat_id: str,
    top_k:   int = FINAL_TOP_K,
) -> str:
    logger.info(f"[RETRIEVER] query={query!r} chat_id={chat_id}")

    section = _infer_section(query)
    fused, graph_context = await _hybrid_fetch(query, chat_id, section)

    if not fused and not graph_context:
        logger.warning("[RETRIEVER] no documents found")
        return ""

    # rerank vector results
    loop    = asyncio.get_running_loop()
    texts   = [d.page_content for d in fused]
    ranked  = await loop.run_in_executor(_POOL, rerank_documents, query, texts, top_k)

    logger.info(f"[RETRIEVER] {len(ranked)} chunks after rerank")

    # build final context — vector chunks + graph context
    parts = []
    if ranked:
        parts.append("### Retrieved Chunks\n" + "\n\n".join(ranked))
    if graph_context:
        parts.append("### Knowledge Graph Context\n" + graph_context)

    return "\n\n".join(parts)


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