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
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_TTL        = 300        # seconds — same query reused for 5 min
CB_FAIL_LIMIT    = 3          # failures before circuit opens
CB_RESET_TIMEOUT = 60         # seconds before circuit half-opens
MIN_RESULT_CHARS = 50
_POOL            = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tavily")

# ── Result cache (query_hash → (result, timestamp)) ──────────────────────────
_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()


# ── Circuit breaker ───────────────────────────────────────────────────────────
class _CircuitBreaker:
    def __init__(self, limit: int, reset: float):
        self._limit, self._reset = limit, reset
        self._fails = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at and time.time() - self._opened_at > self._reset:
                self._fails = 0; self._opened_at = None   # half-open
            return self._opened_at is not None

    def record_success(self):
        with self._lock: self._fails = 0; self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._fails += 1
            if self._fails >= self._limit:
                self._opened_at = time.time()
                logger.warning("[WEB SEARCH] circuit breaker opened")


_cb = _CircuitBreaker(CB_FAIL_LIMIT, CB_RESET_TIMEOUT)


# ── Singleton tool ────────────────────────────────────────────────────────────
class _TavilyTool:
    _instance: Optional[TavilySearchResults] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> TavilySearchResults:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = TavilySearchResults(
                        max_results=settings.TAVILY_MAX_RESULTS
                    )
        return cls._instance


# ── Helpers ───────────────────────────────────────────────────────────────────
def _optimize(query: str) -> str:
    query = re.sub(r"[^\w\s\-]", " ", query.strip())
    query = re.sub(r"\s+", " ", query)
    return f"{query} research paper"

def _clean(results: List[dict]) -> str:
    texts = [
        r["content"].strip() for r in results
        if len(r.get("content", "").strip()) >= MIN_RESULT_CHARS
    ]
    return "\n\n".join(texts[: settings.TAVILY_MAX_RESULTS])

def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().encode()).hexdigest()


# ── Sync core ─────────────────────────────────────────────────────────────────
def run_web_search(query: str) -> str:
    if _cb.is_open():
        logger.warning("[WEB SEARCH] circuit open — skipping")
        return ""

    key = _cache_key(query)
    with _cache_lock:
        if key in _cache:
            result, ts = _cache[key]
            if time.time() - ts < CACHE_TTL:
                logger.info("[WEB SEARCH] cache hit")
                return result

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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_POOL, run_web_search, query)


# ── Cache management ──────────────────────────────────────────────────────────
def clear_search_cache() -> None:
    with _cache_lock: _cache.clear()
