from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import os
import shutil
import asyncio

from src.research_intelligence_system.services.chat_service import process_chat
from api.schemas import ChatRequest, ChatResponse
from src.research_intelligence_system.pipeline.paper_processing_pipeline import ingest_pdf

BASE_PATH = "artifacts/data"
router = APIRouter()


# ---------- CHAT ENDPOINT ----------
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        response = await process_chat(request)   # async

        return ChatResponse(**response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- UPLOAD ENDPOINT ----------
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), chat_id: str = Form(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        chat_path = os.path.join(BASE_PATH, chat_id)
        os.makedirs(chat_path, exist_ok=True)

        file_path = os.path.join(chat_path, file.filename)

        # ---------- SAVE FILE ----------
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---------- ASYNC INGEST ----------
        asyncio.create_task(
            ingest_pdf(chat_id=chat_id, pdf_path=file_path)
        )

        return {
            "status": "processing",
            "message": "PDF uploaded. Ingestion running in background."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))