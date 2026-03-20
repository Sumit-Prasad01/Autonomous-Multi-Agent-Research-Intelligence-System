from __future__ import annotations

import asyncio
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.research_intelligence_system.agents.parsing_agent import ParsingAgent
from src.research_intelligence_system.constants import CHUNK_OVERLAP, CHUNK_SIZE
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MIN_PAGE_CHARS  = 30
MIN_CHUNK_CHARS = 50
_POOL           = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pdf")

# Section separators — coarse → fine, no character-level splits
_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? "]

# Section patterns for metadata tagging
_SECTION_PATTERNS: Dict[str, re.Pattern] = {
    "abstract":     re.compile(r"\babstract\b",              re.I),
    "introduction": re.compile(r"\bintroduction\b",          re.I),
    "methodology":  re.compile(r"\b(method(ology)?|approach)\b", re.I),
    "results":      re.compile(r"\b(results?|findings?)\b",  re.I),
    "discussion":   re.compile(r"\bdiscussion\b",            re.I),
    "conclusion":   re.compile(r"\bconclusion\b",            re.I),
    "references":   re.compile(r"\breferences\b",            re.I),
}

_REF_STRIP = re.compile(
    r"\n?\s*(References|Bibliography|Works Cited)\s*\n.*",
    re.I | re.S,
)


# ── In-process parse cache (keyed by file hash) ───────────────────────────────
_parse_cache: Dict[str, List[Document]] = {}


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Text cleaning ─────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    if not text:
        return ""
    text = _REF_STRIP.sub("", text)             # strip references section
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII noise
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


def _tag_section(text: str) -> str:
    """Return the best-matching section label for a page/chunk."""
    lower = text[:300].lower()
    for label, pat in _SECTION_PATTERNS.items():
        if pat.search(lower):
            return label
    return "body"


# ── Async page loader ─────────────────────────────────────────────────────────
async def _load_pages(pdf_path: str) -> List[Document]:
    """Load raw pages in a threadpool (PyPDFLoader is sync/blocking)."""
    loop = asyncio.get_event_loop()
    docs: List[Document] = await loop.run_in_executor(
        _POOL, lambda: PyPDFLoader(pdf_path).load()
    )
    return docs


async def _clean_page(doc: Document) -> Optional[Document]:
    """Clean a single page async — CPU-bound work offloaded to pool."""
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(_POOL, _clean, doc.page_content)
    if len(text) < MIN_PAGE_CHARS:
        return None
    return Document(
        page_content=text,
        metadata={**doc.metadata, "section": _tag_section(text)},
    )


# ── Public async API ──────────────────────────────────────────────────────────
async def load_pdf_file(pdf_path: str) -> List[Document]:
    """
    Async PDF loader with file-hash cache.
    Returns cleaned, section-tagged Document list.
    """
    if not os.path.exists(pdf_path):
        raise CustomException(f"File not found: {pdf_path}")

    # ── cache check ──────────────────────────────────────────────────────────
    loop   = asyncio.get_event_loop()
    f_hash = await loop.run_in_executor(_POOL, _file_hash, pdf_path)
    if f_hash in _parse_cache:
        logger.info(f"[PDF CACHE HIT] {os.path.basename(pdf_path)}")
        return _parse_cache[f_hash]

    logger.info(f"[PDF LOAD] {pdf_path}")

    raw_pages = await _load_pages(pdf_path)
    if not raw_pages:
        raise CustomException("No content extracted from PDF.")

    # ── parallel page cleaning ────────────────────────────────────────────────
    results = await asyncio.gather(*[_clean_page(p) for p in raw_pages])
    cleaned = [d for d in results if d is not None]

    if not cleaned:
        raise CustomException("All pages were empty after cleaning.")

    logger.info(f"[PDF LOAD] {len(cleaned)}/{len(raw_pages)} pages kept")

    _parse_cache[f_hash] = cleaned
    return cleaned


async def create_text_chunks(
    documents: List[Document],
    chat_id: Optional[str] = None,
) -> List[Document]:
    """
    Async chunking pipeline:
      1. Run ParsingAgent in threadpool (non-blocking)
      2. Split with section-aware separators
      3. Filter short chunks & propagate section metadata
    """
    if not documents:
        raise CustomException("No documents provided for chunking.")

    logger.info(f"[CHUNKING] {len(documents)} pages → parsing …")

    # ── ParsingAgent (sync) in threadpool ────────────────────────────────────
    loop   = asyncio.get_event_loop()
    parser = ParsingAgent()
    parsed: List[Document] = await loop.run_in_executor(
        _POOL, parser.parse_documents, documents
    )

    logger.info(f"[CHUNKING] parsed → splitting …")

    # ── Splitting ────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )

    # run split in threadpool (can be CPU-heavy for large docs)
    raw_chunks: List[Document] = await loop.run_in_executor(
        _POOL, splitter.split_documents, parsed
    )

    # ── Filter + enrich metadata ─────────────────────────────────────────────
    chunks: List[Document] = []
    for c in raw_chunks:
        text = c.page_content.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        meta = {
            **c.metadata,
            "section": c.metadata.get("section") or _tag_section(text),
            "char_count": len(text),
        }
        if chat_id:
            meta["chat_id"] = chat_id
        chunks.append(Document(page_content=text, metadata=meta))

    logger.info(f"[CHUNKING] {len(chunks)} clean chunks generated")
    return chunks


# ── Convenience: load + chunk in one call ────────────────────────────────────
async def parse_pdf(
    pdf_path: str,
    chat_id: Optional[str] = None,
) -> List[Document]:
    """Single entry-point: load → clean → chunk."""
    pages  = await load_pdf_file(pdf_path)
    chunks = await create_text_chunks(pages, chat_id=chat_id)
    return chunks


# ── Cache management ──────────────────────────────────────────────────────────
def clear_parse_cache(pdf_path: Optional[str] = None) -> None:
    if pdf_path:
        h = _file_hash(pdf_path)
        _parse_cache.pop(h, None)
        logger.info(f"Cache cleared for {pdf_path}")
    else:
        _parse_cache.clear()
        logger.info("Full parse cache cleared")
