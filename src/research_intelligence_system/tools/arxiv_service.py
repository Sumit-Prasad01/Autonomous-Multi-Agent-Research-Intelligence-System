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
_CACHE_TTL = 3600
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


# ── Stopwords ─────────────────────────────────────────────────────────────────
_STOPWORDS: frozenset = frozenset({
    # Generic English
    "the", "a", "an", "of", "for", "on", "in", "with", "using", "based",
    "via", "and", "or", "to", "from", "towards", "its", "their", "our",
    "are", "is", "was", "were", "be", "been", "have", "has", "that", "this",
    "which", "than", "more", "also", "each", "both", "all", "any",
    # Research meta-words
    "approach", "method", "model", "paper", "research", "study",
    "analysis", "learning", "system", "framework", "work", "show",
    "propose", "present", "introduce", "demonstrate", "evaluate",
    # Venue names
    "arxiv", "ieee", "acm", "neurips", "icml", "iclr", "emnlp", "acl",
    "naacl", "cvpr", "iccv", "eccv", "aaai", "ijcai", "coling", "interspeech",
    "transactions", "proceedings", "conference", "journal", "workshop",
    # Dataset boilerplate
    "wall", "street", "wsj", "penn", "treebank", "portion",
    "corpus", "dataset", "benchmark", "split", "subset",
    "retrieval", "augmented", "generation",
    # Generic neural architecture fragments
    "layer", "layers", "head", "heads", "scale", "scaled",
    "large", "small", "tiny", "base", "deep", "pre", "fine",
    "step", "block", "blocks", "unit", "units",
    # Generic ML technique names — the primary fix for wrong-domain queries
    "dropout", "regularization", "normalisation", "normalization", "activation",
    "attention", "softmax", "relu", "sigmoid", "tanh", "gelu",
    "pooling", "convolution", "convolutional", "recurrent",
    "backpropagation", "gradient", "gradients", "stochastic",
    "batch", "epoch", "epochs", "loss", "losses",
    "optimization", "optimisation", "momentum", "decay", "weight",
    "positional", "sinusoidal", "encoding", "decoding",
    "encoder", "decoder", "embedding", "embeddings",
    "sequence", "sequences", "token", "tokens",
    "probability", "distribution", "distributions",
    "function", "functions", "mechanism", "mechanisms",
    "dot", "product", "multi", "self", "cross",
    "sparse", "dense", "linear", "nonlinear",
    "hidden", "output", "input", "feed", "forward",
    "training", "testing", "inference", "prediction", "predictions",
    "classification", "detection", "recognition",
    "extraction", "translation",
    "pretraining", "finetuning", "modeling",
    "pretrained", "finetuned", "downstream",
    "supervised", "unsupervised", "semisupervised",
    "multimodal", "multilingual", "multitask", "multiscale",
    # Topic words too broad for arXiv (return entire sub-fields)
    "quantization", "quantisation",   # "quantization" → 50k papers; need model name instead
    "clustering", "compression",
    "approximation",
    # Optimizer names (appear as methods, contaminate queries)
    "adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamw", "nadam", "lars",
    "nesterov",
    # Language / domain words from dataset names
    "english", "french", "german", "chinese", "arabic", "spanish",
    "portuguese", "japanese", "korean", "italian", "dutch",
    "machine", "language", "natural", "text", "image", "visual",
    "speech", "audio", "video", "graph", "tree", "node",
    "word", "words", "sentence", "sentences", "document", "documents",
    "question", "answer", "reading", "comprehension",
    # Generic neural architecture words
    "neural", "network", "networks", "shallow","connection", "connections", "residual", "recurrent"
})


# ── LaTeX subscript artifact cleaner ─────────────────────────────────────────
# When LaTeX like "TurboQuant_{prod}" is stripped by PDF parsing, the subscript
# is concatenated directly: "TurboQuantprod". These known math/stat subscripts
# should be stripped to recover the true entity name "TurboQuant".
_MATH_SUBSCRIPTS: frozenset = frozenset({
    'prod', 'mse', 'mae', 'min', 'max', 'opt', 'est', 'ref',
    'val', 'err', 'acc', 'lat', 'mem', 'out', 'init',
})


def _clean_latex_subscript(name: str) -> str:
    """
    Strip known math/statistics subscript suffixes from entity names.

    These are artifacts of LaTeX PDF parsing where "Model_{subscript}"
    becomes "Modelsubscript" after the braces are stripped.

    Rules:
    - Only operates on names with at least one uppercase letter (entity names).
    - Only strips suffixes from the known math abbreviation set (prod, mse, val...).
    - Root must be ≥ 2 characters after stripping.
    - Name must start with uppercase (otherwise it's a generic word, not an entity).

    Safe by design: 'Base', 'Large', 'Tiny' are NOT in the set, so real model
    variants like "ViTBase", "BERTLarge" are preserved unchanged.

    Examples:
      "TurboQuantprod" → "TurboQuant"
      "TurboQuantmse"  → "TurboQuant"
      "VQprod"         → "VQ"
      "ResNet50"       → "ResNet50"   (no matching suffix)
      "ViTBase"        → "ViTBase"    ('Base' not in set)
    """
    if not name or not name[0].isupper():
        return name  # not an entity name

    for suffix in sorted(_MATH_SUBSCRIPTS, key=len, reverse=True):
        if (name.lower().endswith(suffix)
                and len(name) > len(suffix) + 1):   # root ≥ 2 chars
            return name[:-len(suffix)]

    return name


# ── Token specificity gate ────────────────────────────────────────────────────
def _is_specific_token(tok: str) -> bool:
    """
    Returns True only if a token is specific enough for an arXiv search.

    A token passes if it is:
    1. All-caps acronym ≥ 4 chars: BERT, BLEU, CLIP, DALL (not SLB, NLP, GPT)
       Requiring ≥4 prevents random 2-3 letter abbreviations from appearing.
    2. Title-case proper noun ≥ 4 chars: Transformer, ResNet, ImageNet
    3. Mixed alpha+digit (any length): ResNet50, GPT-4 (after split: GPT4), Llama3
    4. Long specific lowercase ≥ 8 chars, not in stopwords: sinusoidal, flickr30k

    Rejects:
    - Anything in _STOPWORDS (covers generic technique names, lang words, etc.)
    - Short all-caps (≤3): SLB, NLP, GPT (too common in many unrelated contexts)
    - Short generic lowercase: bad heuristics, too broad
    """
    if not tok or not tok[0].isalpha():
        return False

    tl = tok.lower()
    if tl in _STOPWORDS:
        return False

    # All-caps acronym — require ≥4 chars to exclude random 3-letter abbreviations
    if tok.isupper() and len(tok) >= 4:
        return True

    # Title-case proper noun ≥ 4 chars
    if tok[0].isupper() and len(tok) >= 4:
        return True

    # Contains digit mixed with letters: ResNet50, Llama3, GPT4
    if re.search(r'[A-Za-z].*\d|\d.*[A-Za-z]', tok):
        return True

    # Long specific lowercase ≥ 8 chars, not in stopwords
    if tok.islower() and len(tok) >= 8:
        return True

    return False


def _extract_best_token(term: str) -> str:
    """
    Extract the single highest-quality arXiv search token from an entity term.

    Steps:
    1. Clean LaTeX subscript artifacts: "TurboQuantprod" → "TurboQuant"
    2. Strip parenthetical content
    3. Split on whitespace, hyphens, slashes, colons, dots, commas
       (colon split is the fix for "arXiv:2406.03482" — was timing out ArXiv)
    4. Filter: start with letter, pass stopwords + specificity check
    5. Return longest surviving token

    The colon split is the critical fix for arXiv paper IDs appearing in entity
    lists: "arXiv:2406.03482" → ["arXiv", "2406", "03482"] → all filtered.
    Without it: "arXiv:2406.03482" passed as one token and caused 25s timeouts.
    """
    # 1. Clean LaTeX subscript artifacts
    term = _clean_latex_subscript(term)

    # 2. Remove parenthetical content
    clean = re.sub(r'\([^)]*\)', '', term).strip()

    # 3. Split on whitespace, hyphens, slashes, colons, dots, commas, underscores
    tokens = re.split(r'[\s\-/:\.,;_]+', clean)

    # 4. Filter
    quality = [
        w for w in tokens
        if len(w) >= 3
        and w[0].isalpha()
        and w.lower() not in _STOPWORDS
        and not (re.search(r'\d', w) and len(w) < 5)   # skip short numeric "4b", "v2"
        and _is_specific_token(w)
    ]

    if not quality:
        return ""

    # 5. Return longest (more specific within quality set)
    return max(quality, key=len)


class ArxivService:

    async def search(self, query: str, max_results: int = 5) -> List[Dict]:
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
        Build a specific 2–4 token arXiv query from extracted entities.

        All tokens must pass _is_specific_token() and _extract_best_token().
        LaTeX artifacts ("TurboQuantprod") are cleaned before token extraction.
        ArXiv IDs ("arXiv:2406.03482") are now safely split by colon and filtered.
        Generic topic words ("quantization") are blocked by _STOPWORDS.
        Short 2-3 letter acronyms ("SLB") are blocked by the ≥4 char rule.
        """
        parts: List[str] = []

        # Models: up to 2 specific tokens
        for term in models[:6]:
            tok = _extract_best_token(term)
            if tok and tok not in parts:
                parts.append(tok)
            if len(parts) >= 2:
                break

        # Datasets: up to 1 token
        for term in datasets[:4]:
            tok = _extract_best_token(term)
            if tok and tok not in parts:
                parts.append(tok)
                break

        # Methods: up to 1 token — only when we already have other tokens
        if len(parts) >= 1:
            for term in methods[:4]:
                tok = _extract_best_token(term)
                if tok and tok not in parts and len(parts) < 4:
                    parts.append(tok)
                    break

        parts = parts[:4]

        # Fallback: title words if no entity tokens survived
        if not parts and title:
            clean_title = re.sub(r'^[a-f0-9]{32}_', '', title)
            clean_title = re.sub(r'\.pdf$', '', clean_title, flags=re.IGNORECASE)
            clean_title = re.sub(r'[\-_]', ' ', clean_title)
            parts = [
                w for w in clean_title.split()
                if len(w) > 3
                and w[0].isalpha()
                and w.lower() not in _STOPWORDS
                and _is_specific_token(w)
            ][:3]

        if not parts:
            logger.warning("[ARXIV] no specific tokens found — skipping search")
            return []

        query = " ".join(parts).strip()
        logger.info(f"[ARXIV] entity query='{query}'")
        return await self.search(query, max_results)

    async def fetch_paper_details(self, arxiv_id: str) -> Optional[Dict]:
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