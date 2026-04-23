"""
paper_router.py — PDF upload + analysis pipeline endpoints
POST /chats/{chat_id}/upload
GET  /chats/{chat_id}/status
POST /chats/{chat_id}/analyze
GET  /chats/{chat_id}/analysis-status
GET  /chats/{chat_id}/analysis
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import Dict, List, Optional

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from src.research_intelligence_system.constants import BASE_DIR, MAX_MB
from src.research_intelligence_system.database.chat_repository import get_chat
from src.research_intelligence_system.database.database import get_db
from src.research_intelligence_system.database.paper_repository import (
    get_comparison, get_literature_review, get_paper_analyses,
)
from src.research_intelligence_system.pipeline.paper_processing_pipeline import (
    ingest_multiple,
)
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chats", tags=["papers"])

# ── In-memory status stores ───────────────────────────────────────────────────
# These are intentionally module-level so they persist across requests
_ingest_status:   Dict[str, Dict] = {}
_analysis_status: Dict[str, Dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _set_ingest(chat_id: str, status: str, paper_ids: List[str] = [],
                error: Optional[str] = None, pending_paths: List[str] = []) -> None:
    _ingest_status[chat_id] = {
        "status":        status,
        "paper_ids":     paper_ids,
        "error":         error,
        "pending_paths": pending_paths,
    }


def _set_analysis(chat_id: str, status: str, error: Optional[str] = None) -> None:
    _analysis_status[chat_id] = {"status": status, "error": error}


# ── Background tasks ──────────────────────────────────────────────────────────
async def _run_ingest(chat_id: str, file_paths: List[str]) -> None:
    """Ingest PDFs → Qdrant + create PaperAnalysis rows (status=pending)."""
    from src.research_intelligence_system.database.database import AsyncSessionLocal

    _set_ingest(chat_id, "processing")
    logger.info(f"[INGEST] starting chat_id={chat_id} files={len(file_paths)}")

    try:
        async with AsyncSessionLocal() as db:
            paper_ids = await ingest_multiple(chat_id, file_paths, db=db)

        valid_ids = [pid for pid in paper_ids if pid]
        _set_ingest(chat_id, "ready", paper_ids=valid_ids)
        logger.info(f"[INGEST] ready chat_id={chat_id} paper_ids={valid_ids}")

    except Exception as e:
        logger.error(f"[INGEST] failed: {e}", exc_info=True)
        _set_ingest(chat_id, "failed", error=str(e))


async def _run_analysis(chat_id: str, paper_ids: List[str], llm_id: str) -> None:
    """Run full multi-agent analysis pipeline on uploaded papers."""
    from src.research_intelligence_system.agents.orchestrator_agent import OrchestratorAgent
    from src.research_intelligence_system.database.database import AsyncSessionLocal

    _set_analysis(chat_id, "running")
    logger.info(f"[ANALYSIS] starting chat_id={chat_id} papers={paper_ids}")

    try:
        async with AsyncSessionLocal() as db:
            orchestrator = OrchestratorAgent(llm_id=llm_id)
            await orchestrator.run_full_analysis(
                chat_id   = chat_id,
                paper_ids = paper_ids,
                db        = db,
            )
        _set_analysis(chat_id, "complete")
        logger.info(f"[ANALYSIS] complete chat_id={chat_id}")

    except Exception as e:
        logger.error(f"[ANALYSIS] failed: {e}", exc_info=True)
        _set_analysis(chat_id, "failed", error=str(e))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/{chat_id}/upload")
async def upload_pdf(
    chat_id: str,
    file:    UploadFile = File(...),
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB} MB.")

    # save to disk
    chat_dir  = os.path.join(BASE_DIR, chat_id)
    os.makedirs(chat_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    file_path = os.path.join(chat_dir, safe_name)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    logger.info(f"[INGEST] chat_id={chat_id} path={file_path}")

    # accumulate paths for multi-file uploads
    existing = _ingest_status.get(chat_id, {}).get("pending_paths", [])
    existing.append(file_path)
    _set_ingest(chat_id, "queued", pending_paths=existing)

    asyncio.get_event_loop().create_task(_run_ingest(chat_id, existing))
    return {"status": "processing", "filename": safe_name}


@router.get("/{chat_id}/status")
async def ingest_status(
    chat_id: str,
    user:    dict = Depends(get_current_user),
):
    return _ingest_status.get(
        chat_id,
        {"status": "unknown", "paper_ids": [], "error": None},
    )


@router.post("/{chat_id}/analyze")
async def trigger_analysis(
    chat_id: str,
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    """Trigger full multi-agent analysis. Called when user clicks 'Run Analysis'."""
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")

    # get paper_ids from ingest status or Postgres fallback
    status    = _ingest_status.get(chat_id, {})
    paper_ids = status.get("paper_ids", [])

    if not paper_ids:
        analyses  = await get_paper_analyses(db, chat_id)
        paper_ids = [str(a.id) for a in analyses]

    if not paper_ids:
        raise HTTPException(400, "No papers found. Upload a PDF first.")

    if status.get("status") not in ("ready", "complete", ""):
        if status.get("status") == "processing":
            raise HTTPException(400, "PDF still processing — please wait.")

    asyncio.get_event_loop().create_task(
        _run_analysis(chat_id, paper_ids, chat.llm_id)
    )
    return {"status": "running", "paper_count": len(paper_ids)}


@router.get("/{chat_id}/analysis-status")
async def analysis_status(
    chat_id: str,
    user:    dict = Depends(get_current_user),
):
    return _analysis_status.get(
        chat_id,
        {"status": "not_started", "error": None},
    )


@router.get("/{chat_id}/analysis")
async def get_analysis(
    chat_id: str,
    user:    dict = Depends(get_current_user),
    db:      AsyncSession = Depends(get_db),
):
    """Return all agent outputs for this chat."""
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")

    analyses = await get_paper_analyses(db, chat_id)
    if not analyses:
        raise HTTPException(404, "No analysis found. Run analysis first.")

    comparison = await get_comparison(db, chat_id)
    lit_review = await get_literature_review(db, chat_id)

    paper_list = [
        {
            "paper_id":               str(a.id),
            "filename":               a.filename,
            "status":                 a.analysis_status,
            "entities":               a.entities,
            "summaries":              a.summaries,
            "refined_summary":        a.refined_summary,
            "quality_score":          a.quality_score,
            "research_gaps":          a.research_gaps,
            "future_directions":      a.future_directions,
            "triples":                a.triples,
            "similar_papers":         a.similar_papers,
            "hallucination_score":    a.hallucination_score,
            "faithfulness_score":     a.faithfulness_score,
            "hallucinated_sentences": a.hallucinated_sentences,
        }
        for a in analyses
    ]

    return {
        "papers": paper_list,
        "comparison": {
            "comparison_table": comparison.comparison_table  if comparison else {},
            "ranking":          comparison.ranking           if comparison else [],
            "evolution_trends": comparison.evolution_trends  if comparison else "",
            "positioning":      comparison.positioning       if comparison else "",
            "web_papers_used":  comparison.web_papers_used   if comparison else [],
            "comparison_type":  comparison.comparison_type   if comparison else "",
        } if comparison else {},
        "literature_review": {
            "themes":                lit_review.themes                if lit_review else [],
            "review_text":           lit_review.review_text           if lit_review else "",
            "research_gaps_summary": lit_review.research_gaps_summary if lit_review else "",
            "future_directions":     lit_review.future_directions     if lit_review else "",
            "overall_quality":       lit_review.overall_quality       if lit_review else 0.0,
        } if lit_review else {},
        "cross_paper_gaps": {
            "gaps":          lit_review.cross_paper_gaps    if lit_review else [],
            "field_insight": lit_review.field_level_insight if lit_review else "",
            "novelty_score": lit_review.cross_paper_novelty if lit_review else 0.0,
        } if lit_review else {},
    }