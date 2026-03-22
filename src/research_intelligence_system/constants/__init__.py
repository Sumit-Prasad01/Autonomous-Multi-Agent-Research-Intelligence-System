import os
from pathlib import Path

# Config paths
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# api/routes.py
BASE_DIR = "artifacts/data"
MAX_MB   = 50
LLM_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
]

# frontend/ui.py
POLL_INTERVAL = 2.0
MAX_POLLS = 120

# core/qa_system.py
RAG_TIMEOUT = 8
WEB_TIMEOUT = 10
MIN_ANSWER  = 30


# agents/parsing_agent.py
MIN_CHARS = 40

# rag/pdf_parser.py
MIN_PAGE_CHARS  = 30
MIN_CHUNK_CHARS = 50

# rag/vector_store.py
COLLECTION_NAME  = "research_papers"
EMBED_BATCH_SIZE = 64
VECTOR_DIM       = 384          # all-MiniLM-L6-v2 output dim
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# HF_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# VECTOR_DIM    = 768

# rag_retriver.py
DENSE_FETCH_K  = 20          # over-fetch before rerank
BM25_FETCH_K   = 20
FINAL_TOP_K    = 5           # returned to QA system
RRF_K          = 60          # RRF constant (higher = smoother rank blending)

# rag/reranker.py
MODEL_NAME  = "BAAI/bge-reranker-base"
BATCH_SIZE  = 32

# rag/pdf_parser.py
CHUNK_OVERLAP = 200
CHUNK_SIZE = 1000


# services/auth_service.py
ALGORITHM     = "HS256"
ACCESS_TTL    = 60 * 24        # minutes — 1 day

# services/chat_service.py
MAX_HISTORY = 10
MAX_Q_CHARS = 1500

# services/redis_service.py
MAX_MEMORY_MSGS = 10          # messages kept hot in Redis per chat
MSG_TTL         = 60 * 60 * 24  # 24 h — evict idle chats automatically

# tools/web_search.py
CACHE_TTL        = 300        # seconds — same query reused for 5 min
CB_FAIL_LIMIT    = 3          # failures before circuit opens
CB_RESET_TIMEOUT = 60         # seconds before circuit half-opens
MIN_RESULT_CHARS = 50
TAVILY_MAX_RESULTS = 3