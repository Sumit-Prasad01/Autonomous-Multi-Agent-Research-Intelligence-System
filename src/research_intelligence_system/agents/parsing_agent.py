from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from langchain_core.documents import Document

from src.research_intelligence_system.constants import MIN_CHARS
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="parser")

_PATTERNS: Dict[str, re.Pattern] = {
    "abstract":     re.compile(r"\babstract\b",                                     re.I),
    "introduction": re.compile(r"\bintroduction\b",                                 re.I),
    "methodology":  re.compile(r"\b(methodology|methods?|approach|framework)\b",    re.I),
    "results":      re.compile(r"\b(results?|experiments?|evaluation|findings?)\b", re.I),
    "conclusion":   re.compile(r"\b(conclusion|discussion|future\s+work|limitations?)\b", re.I),
    "references":   re.compile(r"\breferences\b",                                   re.I),
}

_NOISE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _NOISE.sub(" ", text.strip())


def _detect_section(header: str, current: str) -> str:
    for label, pat in _PATTERNS.items():
        if pat.search(header[:300]):
            return label
    return current


def _parse_batch(documents: List[Document]) -> List[Document]:
    parsed, current_section = [], "unknown"
    for doc in documents:
        text = _normalize(doc.page_content or "")
        if len(text) < MIN_CHARS:
            continue
        current_section = _detect_section(text, current_section)
        parsed.append(Document(
            page_content=text,
            metadata={**(doc.metadata or {}), "section": current_section},
        ))
    return parsed


class ParsingAgent:

    def parse_documents(self, documents: List[Document]) -> List[Document]:
        return _parse_batch(documents)

    async def async_parse_documents(self, documents: List[Document]) -> List[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_POOL, _parse_batch, documents)