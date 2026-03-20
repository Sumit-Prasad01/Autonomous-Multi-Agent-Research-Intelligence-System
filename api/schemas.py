from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    chat_id: str = Field(..., min_length=1)
    llm_id: str = Field(..., min_length=1)
    allow_search: bool = False
    messages: List[Message]


class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float
    cached: bool = False


class UploadResponse(BaseModel):
    status: str
    message: str
    chat_id: str
    filename: str
