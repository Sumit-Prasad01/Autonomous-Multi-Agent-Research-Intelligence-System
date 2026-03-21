from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.research_intelligence_system.agents.parsing_agent import ParsingAgent
from src.research_intelligence_system.constants import (
    CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS, MIN_PAGE_CHARS
)
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL       = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pdf")
_CACHE_TTL  = 3600   # 1 hour

_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? "]

_SECTION_PATTERNS: Dict[str, re.Pattern] = {
    "abstract":     re.compile(r"\babstract\b",                   re.I),
    "introduction": re.compile(r"\bintroduction\b",               re.I),
    "methodology":  re.compile(r"\b(method(ology)?|approach)\b",  re.I),
    "results":      re.compile(r"\b(results?|findings?)\b",       re.I),
    "discussion":   re.compile(r"\bdiscussion\b",                 re.I),
    "conclusion":   re.compile(r"\bconclusion\b",                 re.I),
    "references":   re.compile(r"\breferences\b",                 re.I),
}

_REF_STRIP = re.compile(
    r"\n?\s*(References|Bibliography|Works Cited)\s*\n.*", re.I | re.S
)

# cache: hash → (docs, timestamp)
_parse_cache: Dict[str, Tuple[List[Document], float]] = {}


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(text: str) -> str:
    if not text: return ""
    text = _REF_STRIP.sub("", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tag_section(text: str) -> str:
    lower = text[:300].lower()
    for label, pat in _SECTION_PATTERNS.items():
        if pat.search(lower):
            return label
    return "body"


async def _load_pages(pdf_path: str) -> List[Document]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_POOL, lambda: PyPDFLoader(pdf_path).load())


async def _clean_page(doc: Document) -> Optional[Document]:
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(_POOL, _clean, doc.page_content)
    if len(text) < MIN_PAGE_CHARS:
        return None
    return Document(
        page_content=text,
        metadata={**doc.metadata, "section": _tag_section(text)},
    )


async def load_pdf_file(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise CustomException(f"File not found: {pdf_path}")

    loop   = asyncio.get_running_loop()
    f_hash = await loop.run_in_executor(_POOL, _file_hash, pdf_path)

    # TTL-based cache check
    entry = _parse_cache.get(f_hash)
    if entry and time.time() - entry[1] < _CACHE_TTL:
        logger.info(f"[PDF CACHE HIT] {os.path.basename(pdf_path)}")
        return entry[0]

    logger.info(f"[PDF LOAD] {pdf_path}")
    raw_pages = await _load_pages(pdf_path)
    if not raw_pages:
        raise CustomException("No content extracted from PDF.")

    results = await asyncio.gather(*[_clean_page(p) for p in raw_pages])
    cleaned = [d for d in results if d is not None]

    if not cleaned:
        raise CustomException("All pages empty after cleaning.")

    logger.info(f"[PDF LOAD] {len(cleaned)}/{len(raw_pages)} pages kept")
    _parse_cache[f_hash] = (cleaned, time.time())
    return cleaned


async def create_text_chunks(
    documents: List[Document],
    chat_id: Optional[str] = None,
) -> List[Document]:
    if not documents:
        raise CustomException("No documents provided for chunking.")

    loop   = asyncio.get_running_loop()
    parser = ParsingAgent()
    parsed = await loop.run_in_executor(_POOL, parser.parse_documents, documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )
    raw_chunks = await loop.run_in_executor(_POOL, splitter.split_documents, parsed)

    chunks = []
    for c in raw_chunks:
        text = c.page_content.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        meta = {
            **c.metadata,
            "section":    c.metadata.get("section") or _tag_section(text),
            "char_count": len(text),
        }
        if chat_id:
            meta["chat_id"] = chat_id
        chunks.append(Document(page_content=text, metadata=meta))

    logger.info(f"[CHUNKING] {len(chunks)} chunks generated")
    return chunks


async def parse_pdf(pdf_path: str, chat_id: Optional[str] = None) -> List[Document]:
    pages  = await load_pdf_file(pdf_path)
    chunks = await create_text_chunks(pages, chat_id=chat_id)
    return chunks


def clear_parse_cache(pdf_path: Optional[str] = None) -> None:
    if pdf_path:
        _parse_cache.pop(_file_hash(pdf_path), None)
    else:
        _parse_cache.clear()
        