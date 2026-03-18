from fastapi import APIRouter, HTTPException

from src.research_intelligence_system.services.chat_service import process_chat
from api.schemas import ChatRequest, ChatResponse, Message

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for QA system.

    Supports:
    - Dynamic LLM selection
    - Optional web search
    - RAG-first answering
    """

    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        response = process_chat(request)

        return ChatResponse(**response)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat request"
        )