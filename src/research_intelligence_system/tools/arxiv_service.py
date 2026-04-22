"""
arxiv_service.py — arXiv API wrapper for fetching related papers
Used by comparison_agent (single paper mode) and orchestrator
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL      = ThreadPoolExecutor(max_workers=2, thread_name_prefix="arxiv")
_CACHE_TTL = 3600   # 1 hour — arXiv results don't change often
_cache: Dict[str, Tuple[List, float]] = {}


def _cache_key(query: str, max_results: int) -> str:
    return hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()


def _get_cached(key: str) -> Optional[List]:
    entry = _cache.get(key)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        return entry[0]
    _cache.pop(key, None)
    return None


def _search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """Sync arXiv search — runs in threadpool."""
    try:
        import arxiv

        search = arxiv.Search(
            query      = query[:100],
            max_results= max_results,
            sort_by    = arxiv.SortCriterion.Relevance,
        )

        papers = []
        for r in search.results():
            papers.append({
                "title":       r.title,
                "abstract":    r.summary[:600],
                "authors":     [a.name for a in r.authors[:4]],
                "year":        r.published.year if r.published else "",
                "arxiv_id":    r.entry_id.split("/")[-1],
                "url":         r.entry_id,
                "categories":  r.categories[:3],
                "source":      "arxiv",
            })

        logger.info(f"[ARXIV] query={query!r:.50} → {len(papers)} papers")
        return papers

    except Exception as e:
        logger.warning(f"[ARXIV] search failed: {e}")
        return []


class ArxivService:

    async def search(
        self,
        query:       str,
        max_results: int = 5,
    ) -> List[Dict]:
        """
        Search arXiv for papers related to query.
        Returns list of paper dicts with title, abstract, authors, year, url.
        """
        if not query or not query.strip():
            return []

        query = query.strip()[:200]
        key   = _cache_key(query, max_results)

        cached = _get_cached(key)
        if cached is not None:
            logger.debug(f"[ARXIV] cache hit query={query!r:.40}")
            return cached

        loop    = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            _POOL, _search_arxiv, query, max_results
        )

        if results:
            _cache[key] = (results, time.time())

        return results

    async def search_by_entities(
        self,
        models:      List[str] = [],
        datasets:    List[str] = [],
        methods:     List[str] = [],
        tasks:       List[str] = [],
        title:       str = "",
        max_results: int = 5,
    ) -> List[Dict]:
        _STOPWORDS = {
            "the", "a", "an", "of", "for", "on", "in", "with",
            "using", "based", "via", "and", "or", "to", "from",
            "towards", "approach", "method", "model", "paper",
            "research", "study", "analysis", "learning",
        }
        parts = []
        for term in models[:2]:
            parts.extend([w for w in term.split() if w.lower() not in _STOPWORDS][:2])
        for term in datasets[:2]:
            parts.extend([w for w in term.split() if w.lower() not in _STOPWORDS][:2])
        for term in methods[:1]:
            parts.extend([w for w in term.split() if w.lower() not in _STOPWORDS][:2])
        for term in tasks[:1]:
            parts.extend([w for w in term.split() if w.lower() not in _STOPWORDS][:1])

        # fallback to title only if nothing else
        if not parts and title:
            parts = [w for w in title.split() if w.lower() not in _STOPWORDS and len(w) > 3][:3]

        parts = list(dict.fromkeys(parts))  # deduplicate preserving order
        query = " ".join(parts).strip()[:80]
        
        if not query:
            return []
        return await self.search(query, max_results)

    async def fetch_paper_details(self, arxiv_id: str) -> Optional[Dict]:
        """Fetch details for a specific arXiv paper by ID."""
        try:
            import arxiv
            loop   = asyncio.get_running_loop()

            def _fetch():
                search = arxiv.Search(id_list=[arxiv_id])
                for r in search.results():
                    return {
                        "title":      r.title,
                        "abstract":   r.summary,
                        "authors":    [a.name for a in r.authors],
                        "year":       r.published.year if r.published else "",
                        "arxiv_id":   arxiv_id,
                        "url":        r.entry_id,
                        "categories": r.categories,
                    }
                return None

            return await loop.run_in_executor(_POOL, _fetch)
        except Exception as e:
            logger.warning(f"[ARXIV] fetch {arxiv_id} failed: {e}")
            return None