from __future__ import annotations

import asyncio
from typing import List

from src.research_intelligence_system.rag.pdf_parser import parse_pdf
from src.research_intelligence_system.rag.vector_store import (
    async_add_documents,
    refresh_vector_store_cache,
)
from src.research_intelligence_system.rag.retriever import invalidate_retriever_cache
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 64


def _batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def ingest_pdf(chat_id: str, pdf_path: str) -> bool:
    """
    Load → clean → chunk → store (batched parallel) → bust caches.
    parse_pdf already handles file-hash caching & async parsing.
    """
    try:
        logger.info(f"[INGEST] start chat_id={chat_id} path={pdf_path}")

        # single await: load + clean + chunk (all async inside parse_pdf)
        chunks = await parse_pdf(pdf_path, chat_id=chat_id)

        if not chunks:
            raise CustomException("No chunks produced from PDF.")

        logger.info(f"[INGEST] {len(chunks)} chunks → storing in batches …")

        # parallel batch writes to vector store
        await asyncio.gather(*[
            async_add_documents(batch, chat_id)
            for batch in _batch(chunks, BATCH_SIZE)
        ])

        # bust BM25 cache for this chat so retriever rebuilds with new docs
        invalidate_retriever_cache(chat_id)

        logger.info(f"[INGEST] done chat_id={chat_id}")
        return True

    except Exception as e:
        logger.exception("[INGEST] pipeline failed")
        raise CustomException("PDF ingestion failed.", e)


async def ingest_multiple(chat_id: str, pdf_paths: List[str]) -> List[bool]:
    """Ingest multiple PDFs for a chat concurrently."""
    return list(await asyncio.gather(
        *[ingest_pdf(chat_id, p) for p in pdf_paths],
        return_exceptions=True,
    ))
