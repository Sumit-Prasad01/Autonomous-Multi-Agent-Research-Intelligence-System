"""
paper_processing_pipeline.py
Stage 1+2: PDF ingestion → chunks → Qdrant
Also creates PaperAnalysis rows in Postgres (status=pending)
Full agent analysis triggered separately via orchestrator
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.research_intelligence_system.rag.pdf_parser import parse_pdf
from src.research_intelligence_system.rag.retriever import invalidate_retriever_cache
from src.research_intelligence_system.rag.vector_store import (
    async_add_documents, invalidate_search_cache
)
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 64


def _batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def _compute_file_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def ingest_pdf(
    chat_id: str,
    pdf_path: str,
    db: Optional[AsyncSession] = None,
) -> str:
    """
    Stage 1+2: Parse PDF → embed → store in Qdrant.
    Creates a PaperAnalysis row in Postgres with status=pending.
    Returns paper_id (UUID string).
    """
    try:
        logger.info(f"[INGEST] chat_id={chat_id} path={pdf_path}")

        # ── parse + chunk ─────────────────────────────────────────────────────
        chunks = await parse_pdf(pdf_path, chat_id=chat_id)
        if not chunks:
            raise CustomException("No chunks produced from PDF.")

        logger.info(f"[INGEST] {len(chunks)} chunks → storing …")

        # ── store in Qdrant ───────────────────────────────────────────────────
        await asyncio.gather(*[
            async_add_documents(batch, chat_id)
            for batch in _batch(chunks, BATCH_SIZE)
        ])

        invalidate_retriever_cache(chat_id)
        invalidate_search_cache(chat_id)

        # ── create PaperAnalysis row in Postgres ──────────────────────────────
        paper_id = None
        if db is not None:
            paper_id = await _create_paper_analysis(
                db=db,
                chat_id=chat_id,
                pdf_path=pdf_path,
            )

        logger.info(f"[INGEST] done chat_id={chat_id} paper_id={paper_id}")
        return paper_id or ""

    except Exception as e:
        logger.exception("[INGEST] failed")
        raise CustomException("PDF ingestion failed.", e)


async def _create_paper_analysis(
    db: AsyncSession,
    chat_id: str,
    pdf_path: str,
) -> str:
    """Create a pending PaperAnalysis row and return its UUID."""
    import uuid as uuid_lib
    import re
    from src.research_intelligence_system.database.models import PaperAnalysis
    from src.research_intelligence_system.database.chat_repository import update_chat_title

    file_hash = await asyncio.get_running_loop().run_in_executor(
        None, _compute_file_hash, pdf_path
    )
    filename = os.path.basename(pdf_path)

    paper = PaperAnalysis(
        id=uuid_lib.uuid4(),
        chat_id=uuid_lib.UUID(chat_id),
        filename=filename,
        file_hash=file_hash,
        analysis_status="pending",
    )
    db.add(paper)
    await db.commit()
    await db.refresh(paper)
    logger.info(f"[INGEST] PaperAnalysis created id={paper.id} status=pending")

    # ── set chat title to cleaned PDF filename ────────────────────────────
    clean = re.sub(r'^[a-f0-9]{32}_', '', filename)  # strip MD5 prefix
    clean = re.sub(r'\.pdf$', '', clean, flags=re.IGNORECASE)
    clean = clean.replace("_", " ").replace("-", " ").strip()
    if clean:
        await update_chat_title(db, chat_id, clean[:100])
        logger.info(f"[INGEST] chat title set to '{clean[:100]}'")

    return str(paper.id)

async def ingest_multiple(
    chat_id: str,
    pdf_paths: List[str],
    db: Optional[AsyncSession] = None,
) -> List[str]:
    """
    Ingest multiple PDFs concurrently.
    Returns list of paper_ids.
    """
    results = await asyncio.gather(
        *[ingest_pdf(chat_id, p, db=db) for p in pdf_paths],
        return_exceptions=True,
    )
    paper_ids = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"[INGEST] failed for {pdf_paths[i]}: {r}")
            paper_ids.append("")
        else:
            paper_ids.append(r)
    return paper_ids