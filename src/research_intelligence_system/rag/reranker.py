"""
reranker.py — Async, batched, cached cross-encoder reranker
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "BAAI/bge-reranker-base"
BATCH_SIZE  = 32
_POOL       = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")


# ── Singleton model ───────────────────────────────────────────────────────────
class _RerankerModel:
    _instance: CrossEncoder | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> CrossEncoder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Loading reranker model …")
                    cls._instance = CrossEncoder(MODEL_NAME)
        return cls._instance


# ── Core (sync, batched) ──────────────────────────────────────────────────────
def rerank_documents(query: str, docs: List[str], top_k: int = 5) -> List[str]:
    if not docs:
        return []
    if len(docs) <= top_k:
        return docs

    model = _RerankerModel.get()
    pairs: List[Tuple[str, str]] = [(query, d) for d in docs]

    # batch predict to avoid OOM on large candidate sets
    scores = []
    for i in range(0, len(pairs), BATCH_SIZE):
        scores.extend(model.predict(pairs[i : i + BATCH_SIZE]))

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ── Async wrapper ─────────────────────────────────────────────────────────────
async def async_rerank(query: str, docs: List[str], top_k: int = 5) -> List[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_POOL, rerank_documents, query, docs, top_k)
