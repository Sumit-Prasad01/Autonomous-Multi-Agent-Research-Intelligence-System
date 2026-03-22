from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from sentence_transformers import CrossEncoder

from src.research_intelligence_system.constants import BATCH_SIZE, MODEL_NAME
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL      = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")
_CACHE_TTL = 120   # 2 min rerank cache

# rerank cache: key → (results, timestamp)
_cache: Dict[str, Tuple[List[str], float]] = {}
_cache_lock = threading.Lock()


def _cache_key(query: str, docs: List[str], top_k: int) -> str:
    raw = f"{query}:{','.join(docs[:5])}:{top_k}"
    return hashlib.md5(raw.encode()).hexdigest()


class _RerankerModel:
    _instance: CrossEncoder | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> CrossEncoder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Loading reranker model …")
                    cls._instance = CrossEncoder(MODEL_NAME, local_files_only=True)
        return cls._instance


def rerank_documents(query: str, docs: List[str], top_k: int = 5) -> List[str]:
    if not docs:     return []
    if len(docs) <= top_k: return docs

    key = _cache_key(query, docs, top_k)
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry[1] < _CACHE_TTL:
            logger.debug("[RERANK CACHE HIT]")
            return entry[0]

    model  = _RerankerModel.get()
    pairs  = [(query, d) for d in docs]
    scores = []
    for i in range(0, len(pairs), BATCH_SIZE):
        scores.extend(model.predict(pairs[i: i + BATCH_SIZE]))

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    result = [doc for doc, _ in ranked[:top_k]]

    with _cache_lock:
        _cache[key] = (result, time.time())
    return result


async def async_rerank(query: str, docs: List[str], top_k: int = 5) -> List[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_POOL, rerank_documents, query, docs, top_k)