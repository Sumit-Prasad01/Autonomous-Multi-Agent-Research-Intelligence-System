from __future__ import annotations

import asyncio
import os
import uuid
from typing import Dict, List

import aiofiles
from fastapi import (APIRouter, Depends, HTTPException, UploadFile, File)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    ChatCreateRequest, ChatMessageRequest, ChatUpdateRequest,
    LoginRequest, RegisterRequest,
)
from src.research_intelligence_system.constants import BASE_DIR, MAX_MB
from src.research_intelligence_system.database.chat_repository import (
    create_chat, delete_chat, get_chat, get_user_chats, update_chat_title
)
from src.research_intelligence_system.database.database import get_db, check_db_health
from src.research_intelligence_system.database.paper_repository import (
    get_paper_analyses, get_paper_analysis
)
from src.research_intelligence_system.pipeline.paper_processing_pipeline import (
    ingest_pdf, ingest_multiple
)
from src.research_intelligence_system.rag.retriever import invalidate_retriever_cache
from src.research_intelligence_system.services.auth_service import (
    decode_token, login_user, logout_user, register_user
)
from src.research_intelligence_system.services.chat_service import (
    delete_chat_session, load_chat_history, process_chat, stream_chat
)
from src.research_intelligence_system.services.redis_service import (
    check_redis_health, create_session, get_session, delete_session
)
from src.research_intelligence_system.database.paper_repository import (
    get_paper_analyses, get_paper_analysis,
    get_comparison, get_literature_review,
)
from src.research_intelligence_system.database.paper_repository import get_paper_code
from src.research_intelligence_system.utils.logger import get_logger

logger   = get_logger(__name__)
router   = APIRouter(prefix="/api/v1", tags=["api"])
security = HTTPBearer()

# ── In-memory status stores ───────────────────────────────────────────────────
_ingest_status:   Dict[str, Dict] = {}   # chat_id → {status, paper_ids, error}
_analysis_status: Dict[str, Dict] = {}   # chat_id → {status, error}


# ── Auth dependency ───────────────────────────────────────────────────────────
async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    payload = await decode_token(creds.credentials)
    if not payload:
        raise HTTPException(401, "Invalid or expired token.")
    return payload


# ── Auth ──────────────────────────────────────────────────────────────────────
@router.post("/auth/register", status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await register_user(db, req.email, req.username, req.password)
        return {"id": str(user.id), "email": user.email, "username": user.username}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/auth/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    token = await login_user(db, req.email, req.password)
    if not token:
        raise HTTPException(401, "Invalid credentials.")
    session_id = await create_session(token, req.email)
    return {"access_token": token, "token_type": "bearer", "session_id": session_id}


@router.post("/auth/logout")
async def logout(creds: HTTPAuthorizationCredentials = Depends(security)):
    await logout_user(creds.credentials)
    return {"message": "Logged out."}


# ── Session ───────────────────────────────────────────────────────────────────
@router.get("/auth/session/{session_id}")
async def restore_session(session_id: str):
    data = await get_session(session_id)
    if not data:
        raise HTTPException(401, "Session expired.")
    return data


@router.delete("/auth/session/{session_id}")
async def remove_session(session_id: str):
    await delete_session(session_id)
    return {"message": "Session deleted."}


# ── Chat CRUD ─────────────────────────────────────────────────────────────────
@router.post("/chats", status_code=201)
async def create_new_chat(
    req: ChatCreateRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await create_chat(db, user["sub"], req.title, req.llm_id, req.allow_search)
    return {"id": str(chat.id), "title": chat.title,
            "llm_id": chat.llm_id, "created_at": str(chat.created_at)}


@router.get("/chats")
async def list_chats(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chats = await get_user_chats(db, user["sub"])
    return [{"id": str(c.id), "title": c.title,
             "llm_id": c.llm_id, "updated_at": str(c.updated_at)}
            for c in chats]


@router.get("/chats/{chat_id}/history")
async def get_history(
    chat_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    return await load_chat_history(chat_id, db)


@router.patch("/chats/{chat_id}")
async def rename_chat(
    chat_id: str,
    req: ChatUpdateRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    await update_chat_title(db, chat_id, req.title)
    return {"message": "Updated."}


@router.delete("/chats/{chat_id}")
async def remove_chat(
    chat_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ok = await delete_chat(db, chat_id, user["sub"])
    if not ok:
        raise HTTPException(404, "Chat not found.")
    await delete_chat_session(chat_id)
    invalidate_retriever_cache(chat_id)
    return {"message": "Deleted."}


# ── Messaging ─────────────────────────────────────────────────────────────────
@router.post("/chats/{chat_id}/message")
async def send_message(
    chat_id: str,
    req: ChatMessageRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    return await process_chat(
        chat_id=chat_id,
        user_message=req.message,
        llm_id=req.llm_id,
        allow_search=req.allow_search,
        db=db,
    )


@router.post("/chats/{chat_id}/stream")
async def stream_message(
    chat_id: str,
    req: ChatMessageRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")

    async def _events():
        try:
            async for token in stream_chat(
                chat_id=chat_id,
                user_message=req.message,
                llm_id=req.llm_id,
                allow_search=req.allow_search,
                db=db,
            ):
                if token:
                    yield f"data: {token.replace(' ', chr(160))}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


# ── PDF upload ────────────────────────────────────────────────────────────────
async def _run_ingest(chat_id: str, file_paths: List[str]):
    """
    Ingest one or more PDFs.
    Creates PaperAnalysis rows with status=pending.
    Stores paper_ids in _ingest_status for analysis trigger.
    """
    from src.research_intelligence_system.database.database import AsyncSessionLocal

    _ingest_status[chat_id] = {"status": "processing", "paper_ids": [], "error": None}
    try:
        logger.info(f"[INGEST] starting chat_id={chat_id} files={len(file_paths)}")
        async with AsyncSessionLocal() as db:
            paper_ids = await ingest_multiple(chat_id, file_paths, db=db)

        _ingest_status[chat_id] = {
            "status":    "ready",
            "paper_ids": paper_ids,
            "error":     None,
        }
        logger.info(f"[INGEST] ready chat_id={chat_id} paper_ids={paper_ids}")
    except Exception as e:
        _ingest_status[chat_id] = {"status": "failed", "paper_ids": [], "error": str(e)}
        logger.error(f"[INGEST] failed: {e}", exc_info=True)


@router.post("/chats/{chat_id}/upload")
async def upload_pdf(
    chat_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB} MB.")

    chat_dir  = os.path.join(BASE_DIR, chat_id)
    os.makedirs(chat_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    file_path = os.path.join(chat_dir, safe_name)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # accumulate file paths if multiple uploads in same chat
    existing = _ingest_status.get(chat_id, {}).get("pending_paths", [])
    existing.append(file_path)
    _ingest_status[chat_id] = {"status": "queued", "pending_paths": existing, "paper_ids": [], "error": None}

    asyncio.get_event_loop().create_task(_run_ingest(chat_id, existing))
    return {"status": "processing", "filename": safe_name}


@router.get("/chats/{chat_id}/status")
async def ingest_status(
    chat_id: str,
    user: dict = Depends(get_current_user),
):
    return _ingest_status.get(chat_id, {"status": "unknown", "paper_ids": [], "error": None})


# ── Analysis (orchestrator trigger) ──────────────────────────────────────────
async def _run_analysis(chat_id: str, paper_ids: List[str], llm_id: str):
    """Run full multi-agent analysis on uploaded papers."""
    from src.research_intelligence_system.database.database import AsyncSessionLocal
    from src.research_intelligence_system.agents.orchestrator_agent import OrchestratorAgent

    _analysis_status[chat_id] = {"status": "running", "error": None}
    try:
        logger.info(f"[ANALYSIS] starting chat_id={chat_id} papers={paper_ids}")
        async with AsyncSessionLocal() as db:
            orchestrator = OrchestratorAgent(llm_id=llm_id)
            await orchestrator.run_full_analysis(
                chat_id=chat_id,
                paper_ids=paper_ids,
                db=db,
            )
        _analysis_status[chat_id] = {"status": "complete", "error": None}
        logger.info(f"[ANALYSIS] complete chat_id={chat_id}")
    except Exception as e:
        _analysis_status[chat_id] = {"status": "failed", "error": str(e)}
        logger.error(f"[ANALYSIS] failed: {e}", exc_info=True)


@router.post("/chats/{chat_id}/analyze")
async def trigger_analysis(
    chat_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Triggered when user clicks 'Get Analysis' button.
    Runs orchestrator → all agents → saves results.
    """
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")

    status = _ingest_status.get(chat_id, {})
    if status.get("status") != "ready":
        raise HTTPException(400, "Papers not ready. Upload and wait for ingestion to complete.")

    paper_ids = status.get("paper_ids", [])
    if not paper_ids:
        # try fetching from Postgres if server restarted
        analyses = await get_paper_analyses(db, chat_id)
        paper_ids = [str(a.id) for a in analyses]

    if not paper_ids:
        raise HTTPException(400, "No papers found for this chat.")

    asyncio.get_event_loop().create_task(
        _run_analysis(chat_id, paper_ids, chat.llm_id)
    )
    return {"status": "running", "paper_count": len(paper_ids)}


@router.get("/chats/{chat_id}/analysis-status")
async def analysis_status(
    chat_id: str,
    user: dict = Depends(get_current_user),
):
    return _analysis_status.get(chat_id, {"status": "not_started", "error": None})

async def _get_code_safe(db, paper_id: str) -> dict:
    try:
        code = await get_paper_code(db, paper_id)
        if not code:
            return {}
        return {
            "algorithm_steps":  code.algorithm_steps,
            "pseudocode":       code.pseudocode,
            "python_skeleton":  code.python_skeleton,
            "time_complexity":  code.time_complexity,
            "space_complexity": code.space_complexity,
            "key_components":   code.key_components,
        }
    except Exception:
        return {}


@router.get("/chats/{chat_id}/analysis")
async def get_analysis(
    chat_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all agent outputs for this chat."""
    chat = await get_chat(db, chat_id, user["sub"])
    if not chat:
        raise HTTPException(404, "Chat not found.")

    analyses = await get_paper_analyses(db, chat_id)
    if not analyses:
        raise HTTPException(404, "No analysis found. Run 'Get Analysis' first.")

    comparison = await get_comparison(db, chat_id)
    lit_review = await get_literature_review(db, chat_id)

    # build papers list with await for code (can't use await in list comprehension)
    paper_list = []
    for a in analyses:
        code = await _get_code_safe(db, str(a.id))
        paper_list.append({
            "paper_id":          str(a.id),
            "filename":          a.filename,
            "status":            a.analysis_status,
            "entities":          a.entities,
            "summaries":         a.summaries,
            "refined_summary":   a.refined_summary,
            "quality_score":     a.quality_score,
            "research_gaps":     a.research_gaps,
            "future_directions": a.future_directions,
            "triples":           a.triples,
            "similar_papers":    a.similar_papers,
            "code":              code,
        })

    return {
        "papers": paper_list,
        "comparison": {
            "comparison_table":  comparison.comparison_table  if comparison else {},
            "ranking":           comparison.ranking           if comparison else [],
            "evolution_trends":  comparison.evolution_trends  if comparison else "",
            "positioning":       comparison.positioning       if comparison else "",
            "web_papers_used":   comparison.web_papers_used   if comparison else [],
            "comparison_type":   comparison.comparison_type   if comparison else "",
        } if comparison else {},
        "literature_review": {
            "themes":                lit_review.themes                if lit_review else [],
            "review_text":           lit_review.review_text           if lit_review else "",
            "research_gaps_summary": lit_review.research_gaps_summary if lit_review else "",
            "future_directions":     lit_review.future_directions     if lit_review else "",
            "overall_quality":       lit_review.overall_quality       if lit_review else 0.0,
        } if lit_review else {},
    }


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    db_ok    = await check_db_health()
    redis_ok = await check_redis_health()
    return {
        "status":   "ok" if db_ok and redis_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
        "redis":    "ok" if redis_ok else "unreachable",
    }