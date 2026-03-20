from __future__ import annotations

import asyncio
import os
import uuid
from typing import Dict

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, UploadResponse
from src.research_intelligence_system.services.chat_service import (
    process_chat,
    stream_chat,
)
from src.research_intelligence_system.pipeline.paper_processing_pipeline import ingest_pdf
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

router   = APIRouter(prefix="/api/v1", tags=["rag"])
BASE_DIR = "artifacts/data"

# ── Ingestion status store (chat_id → status dict) ───────────────────────────
_ingest_status: Dict[str, Dict] = {}

MAX_PDF_MB   = 50
ALLOWED_MIME = {"application/pdf"}


# ── Background ingestion wrapper ──────────────────────────────────────────────
async def _run_ingest(chat_id: str, file_path: str) -> None:
    _ingest_status[chat_id] = {"status": "processing", "error": None}
    try:
        await ingest_pdf(chat_id=chat_id, pdf_path=file_path)
        _ingest_status[chat_id] = {"status": "ready", "error": None}
        logger.info(f"[INGEST] done chat_id={chat_id}")
    except Exception as e:
        _ingest_status[chat_id] = {"status": "failed", "error": str(e)}
        logger.error(f"[INGEST] failed chat_id={chat_id}: {e}")


# ── Chat ──────────────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(400, "messages cannot be empty")
    try:
        result = await process_chat(request)
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("[/chat] error")
        raise HTTPException(500, str(e))


# ── Streaming chat (SSE) ──────────────────────────────────────────────────────
@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    if not request.messages:
        raise HTTPException(400, "messages cannot be empty")

    async def _event_stream():
        try:
            async for token in stream_chat(request):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ── Upload ────────────────────────────────────────────────────────────────────
@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chat_id: str = Form(...),
) -> UploadResponse:

    # ── validation ────────────────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    content = await file.read()
    if len(content) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_PDF_MB} MB limit")
    await file.seek(0)

    # ── save with async I/O ───────────────────────────────────────────────────
    chat_dir  = os.path.join(BASE_DIR, chat_id)
    os.makedirs(chat_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    file_path = os.path.join(chat_dir, safe_name)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    logger.info(f"[UPLOAD] saved {file_path}")

    # ── background ingestion ──────────────────────────────────────────────────
    background_tasks.add_task(_run_ingest, chat_id, file_path)

    return UploadResponse(
        status="processing",
        message="PDF uploaded. Ingestion running in background.",
        chat_id=chat_id,
        filename=safe_name,
    )


# ── Ingestion status ──────────────────────────────────────────────────────────
@router.get("/status/{chat_id}")
async def ingest_status(chat_id: str) -> Dict:
    return _ingest_status.get(chat_id, {"status": "unknown", "error": None})


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health")
async def health() -> Dict:
    return {"status": "ok"}
