from pydantic import BaseModel, Field
from typing import List



class Message(BaseModel):
    content: str = Field(..., min_length=1)



class ChatRequest(BaseModel):
    llm_id: str = Field(..., min_length=1)
    allow_search: bool = False
    messages: List[Message]



class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float