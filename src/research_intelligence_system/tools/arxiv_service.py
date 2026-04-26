"""
arxiv_service.py — arXiv API wrapper for fetching related papers
Used by comparison_agent (single paper mode) and orchestrator
"""
from __future__ import annotations

import re
import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL      = ThreadPoolExecutor(max_workers=2, thread_name_prefix="arxiv")
_CACHE_TTL = 3600   # 1 hour
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
    """Sync arXiv search — 25s timeout enforced by caller, 1 retry max."""
    import arxiv

    clean_query = re.sub(r'[/\\%]', ' ', query)
    clean_query = re.sub(r'\s{2,}', ' ', clean_query).strip()[:80]

    try:
        client = arxiv.Client(
            page_size     = max_results,
            delay_seconds = 3,
            num_retries   = 1,
        )
        search = arxiv.Search(
            query       = clean_query,
            max_results = max_results,
            sort_by     = arxiv.SortCriterion.Relevance,
        )

        papers = []
        for r in client.results(search):
            papers.append({
                "title":      r.title,
                "abstract":   r.summary[:600],
                "authors":    [a.name for a in r.authors[:4]],
                "year":       r.published.year if r.published else "",
                "arxiv_id":   r.entry_id.split("/")[-1],
                "url":        r.entry_id,
                "categories": r.categories[:3],
                "source":     "arxiv",
            })

        logger.info(f"[ARXIV] query='{clean_query[:60]}' → {len(papers)} papers")
        return papers

    except Exception as e:
        logger.warning(f"[ARXIV] search failed: {e}")
        return []


# ── Stopwords for query building ──────────────────────────────────────────────
# Covers: generic English, venue names, dataset boilerplate, structural
# compound-word fragments (layer, head) that are meaningless alone.
_STOPWORDS = {
    # generic English
    "the", "a", "an", "of", "for", "on", "in", "with", "using", "based",
    "via", "and", "or", "to", "from", "towards", "its", "their", "our",
    # meta words
    "approach", "method", "model", "paper", "research", "study",
    "analysis", "learning", "system", "framework",
    # venue names
    "arxiv", "ieee", "acm", "neurips", "icml", "iclr", "emnlp", "acl",
    "naacl", "cvpr", "iccv", "eccv", "aaai", "ijcai",
    "transactions", "proceedings", "conference", "journal", "workshop",
    # dataset boilerplate
    "retrieval", "augmented", "generation",
    "wall", "street", "wsj", "penn", "treebank", "portion",
    "corpus", "dataset", "benchmark", "split", "subset",
    # generic structural fragments from hyphenated compound names
    # e.g. "4-layer" → ["4","layer"] — "layer" alone is noise
    "layer", "layers", "head", "heads", "scale", "scaled",
    "large", "small", "tiny", "base", "deep", "pre", "fine",
    "step", "block", "blocks", "unit", "units",
}


def _extract_best_token(term: str) -> str:
    """
    Extract the single highest-quality search token from an entity term.

    Strategy:
    1. Strip parenthetical abbreviations: "Wall Street Journal (WSJ)" → "Wall Street Journal"
    2. Split on BOTH whitespace AND hyphens — fixes "4-layer" → ["4","layer"]
    3. Filter: must start with a letter, pass stopword check, length ≥ 3
    4. Sort remaining tokens by length descending → longer = more specific
    5. Return the longest quality token, or "" if none found.

    Examples:
      "4-layer transformer"           → "transformer"   (not "-layer")
      "sinusoidal positional encoding"→ "sinusoidal"
      "Wall Street Journal (WSJ)..."  → ""              (all tokens are stopwords)
      "multi-head attention"          → "attention"
      "ResNet50"                      → "ResNet50"
      "ViT-B/16"                      → "ViT"
    """
    # 1. Remove parenthetical content
    clean = re.sub(r'\([^)]*\)', '', term).strip()
    # 2. Split on whitespace and hyphens and slashes
    tokens = re.split(r'[\s\-/]+', clean)
    # 3. Filter quality tokens
    quality = [
        w for w in tokens
        if len(w) >= 3
        and w[0].isalpha()                          # must start with letter (no "-layer", "(WSJ)")
        and w.lower() not in _STOPWORDS
        and not (re.search(r'\d', w) and len(w) < 5)  # skip short numeric tokens like "4b"
    ]
    # 4. Return longest (most specific) token
    if not quality:
        return ""
    return max(quality, key=len)


class ArxivService:

    async def search(
        self,
        query:       str,
        max_results: int = 5,
    ) -> List[Dict]:
        if not query or not query.strip():
            return []

        query = query.strip()[:200]
        key   = _cache_key(query, max_results)

        cached = _get_cached(key)
        if cached is not None:
            logger.debug(f"[ARXIV] cache hit query={query!r:.40}")
            return cached

        loop = asyncio.get_running_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(_POOL, _search_arxiv, query, max_results),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[ARXIV] timeout after 25s query='{query[:50]}'")
            results = []
        except Exception as e:
            logger.warning(f"[ARXIV] executor error: {e}")
            results = []

        # cache both hits (1hr) and misses (5min)
        ttl = _CACHE_TTL if results else 300
        _cache[key] = (results, time.time() - (_CACHE_TTL - ttl))

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
        """
        Build a clean 3–4 token arXiv query from extracted entities.

        Token extraction rules:
        - One best token per entity term (longest quality token after hyphen-split).
        - Model terms take priority (up to 2 tokens).
        - Dataset terms next (up to 1 token, hard to get useful signal from names).
        - Method terms last (up to 1 token).
        - Hard cap: 4 tokens total.
        - Fallback to paper title if no entity tokens survive the filter.
        """
        parts: List[str] = []

        # ── Models: up to 2 tokens ────────────────────────────────────────────
        for term in models[:4]:          # sample more, filter aggressively
            tok = _extract_best_token(term)
            if tok and tok not in parts:
                parts.append(tok)
            if len(parts) >= 2:
                break

        # ── Datasets: up to 1 token ───────────────────────────────────────────
        for term in datasets[:3]:
            tok = _extract_best_token(term)
            if tok and tok not in parts:
                parts.append(tok)
                break                    # one dataset token is enough

        # ── Methods: up to 1 token ────────────────────────────────────────────
        for term in methods[:2]:
            tok = _extract_best_token(term)
            if tok and tok not in parts:
                parts.append(tok)
                break

        # ── Hard cap ─────────────────────────────────────────────────────────
        parts = parts[:4]

        # ── Fallback: use title if nothing survived ───────────────────────────
        if not parts and title:
            clean_title = re.sub(r'^[a-f0-9]{32}_', '', title)
            clean_title = re.sub(r'\.pdf$', '', clean_title, flags=re.IGNORECASE)
            clean_title = re.sub(r'[\-_]', ' ', clean_title)
            parts = [
                w for w in clean_title.split()
                if len(w) > 3
                and w[0].isalpha()
                and w.lower() not in _STOPWORDS
            ][:3]

        if not parts:
            return []

        query = " ".join(parts).strip()
        logger.info(f"[ARXIV] entity query='{query}'")
        return await self.search(query, max_results)

    async def fetch_paper_details(self, arxiv_id: str) -> Optional[Dict]:
        """Fetch details for a specific arXiv paper by ID."""
        try:
            import arxiv
            loop = asyncio.get_running_loop()

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