from pydantic import BaseModel, Field, EmailStr
from typing import List, Literal, Optional

# ── Schemas ───────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class ChatCreateRequest(BaseModel):
    title:        str  = "New Chat"
    llm_id:       str  = "llama-3.1-8b-instant"
    allow_search: bool = False

class ChatMessageRequest(BaseModel):
    message:      str
    llm_id:       str  = "llama-3.1-8b-instant"
    allow_search: bool = False

class ChatUpdateRequest(BaseModel):
    title: str