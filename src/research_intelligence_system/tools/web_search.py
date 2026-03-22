from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.constants import (
    CACHE_TTL, CB_FAIL_LIMIT, CB_RESET_TIMEOUT, MIN_RESULT_CHARS, TAVILY_MAX_RESULTS
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL       = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tavily")
_cache:     Dict[str, tuple] = {}
_cache_lock = threading.Lock()


# ── Circuit breaker ───────────────────────────────────────────────────────────
class _CircuitBreaker:
    def __init__(self, limit: int, reset: float):
        self._limit   = limit
        self._reset   = reset
        self._fails   = 0
        self._opened_at: Optional[float] = None
        self._lock    = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at and time.time() - self._opened_at > self._reset:
                self._fails = 0
                self._opened_at = None
            return self._opened_at is not None

    def record_success(self):
        with self._lock:
            self._fails = 0
            self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._fails += 1
            if self._fails >= self._limit:
                self._opened_at = time.time()
                logger.warning("[WEB SEARCH] circuit breaker opened")


_cb = _CircuitBreaker(CB_FAIL_LIMIT, CB_RESET_TIMEOUT)


# ── Singleton Tavily tool ─────────────────────────────────────────────────────
class _TavilyTool:
    _instance: Optional[TavilySearchResults] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> TavilySearchResults:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = TavilySearchResults(
                        max_results=TAVILY_MAX_RESULTS
                    )
        return cls._instance


# ── Helpers ───────────────────────────────────────────────────────────────────
def _optimize(query: str) -> str:
    query = re.sub(r"[^\w\s\-]", " ", query.strip())
    return re.sub(r"\s+", " ", query) + " research paper"


def _clean(results) -> str:
    if isinstance(results, str):
        return results
    if not isinstance(results, list):
        return ""
    texts = [
        r["content"].strip() for r in results
        if isinstance(r, dict) and len(r.get("content", "").strip()) >= MIN_RESULT_CHARS
    ]
    return "\n\n".join(texts[: TAVILY_MAX_RESULTS])


def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().encode()).hexdigest()


# ── Sync core ─────────────────────────────────────────────────────────────────
def run_web_search(query: str) -> str:
    if _cb.is_open():
        logger.warning("[WEB SEARCH] circuit open — skipping")
        return ""

    key = _cache_key(query)
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry[1] < CACHE_TTL:
            logger.info("[WEB SEARCH] cache hit")
            return entry[0]

    try:
        logger.info(f"[WEB SEARCH] query={query!r}")
        results = _TavilyTool.get().invoke({"query": _optimize(query)})
        text    = _clean(results) if results else ""
        _cb.record_success()
        with _cache_lock:
            _cache[key] = (text, time.time())
        return text
    except Exception as e:
        _cb.record_failure()
        logger.error(f"[WEB SEARCH] failed: {e}")
        return ""


# ── Async wrapper ─────────────────────────────────────────────────────────────
async def async_web_search(query: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_POOL, run_web_search, query)


# ── Cache management ──────────────────────────────────────────────────────────
def clear_search_cache() -> None:
    with _cache_lock:
        _cache.clear()