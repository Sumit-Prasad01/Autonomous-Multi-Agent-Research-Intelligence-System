from __future__ import annotations

import os, threading
import asyncio
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter,
    MatchValue, PayloadSchemaType, VectorParams,
)
from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    Distance, FieldCondition, Filter,
    MatchValue, PayloadSchemaType, VectorParams, PointStruct 
)
import uuid 

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.constants import (
    COLLECTION_NAME, EMBED_BATCH_SIZE, HF_MODEL_NAME, VECTOR_DIM
)
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger


logger = get_logger(__name__)

_POOL        = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qdrant")
_CACHE_TTL   = 60
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0

_search_cache: Dict[str, Tuple[list, float]] = {}
_cache_lock   = threading.Lock()


# ── Search cache ──────────────────────────────────────────────────────────────
def _cache_key(query: str, k: int, chat_id: Optional[str]) -> str:
    return hashlib.md5(f"{query}:{k}:{chat_id}".encode()).hexdigest()

def _get_cached(key: str) -> Optional[list]:
    with _cache_lock:
        entry = _search_cache.get(key)
        if entry and time.time() - entry[1] < _CACHE_TTL:
            return entry[0]
        _search_cache.pop(key, None)
        return None

def _set_cache(key: str, value: list) -> None:
    with _cache_lock:
        _search_cache[key] = (value, time.time())

def invalidate_search_cache(chat_id: Optional[str] = None) -> None:
    with _cache_lock:
        if chat_id:
            for k in [k for k in _search_cache if chat_id in k]:
                del _search_cache[k]
        else:
            _search_cache.clear()


# ── GPU Embeddings ────────────────────────────────────────────────────────────
class FastGPUEmbeddings(Embeddings):
    """Direct sentence-transformers — actually uses GPU unlike LangChain wrapper."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model on {self.device.upper()} …")
        self.model = SentenceTransformer(
            HF_MODEL_NAME,
            device=self.device,
            local_files_only=True,
        )
        logger.info("Embedding model ready.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            device=self.device,
        ).tolist()


# ── Qdrant filter ─────────────────────────────────────────────────────────────
def _chat_filter(chat_id: str) -> Filter:
    return Filter(must=[FieldCondition(
        key="metadata.chat_id", match=MatchValue(value=chat_id)
    )])


# ── Singleton manager ─────────────────────────────────────────────────────────
class VectorStoreManager:
    _instance: Optional["VectorStoreManager"] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self):
        logger.info(f"[VS] init PID={os.getpid()} thread={threading.current_thread().name}")
        self._embeddings: Optional[FastGPUEmbeddings]  = None
        self._client:     Optional[QdrantClient]       = None
        self._db:         Optional[QdrantVectorStore]  = None
        self._lock = threading.Lock()

    @property
    def embeddings(self) -> FastGPUEmbeddings:
        if not self._embeddings:
            with self._lock:
                if not self._embeddings:
                    self._embeddings = FastGPUEmbeddings()
        return self._embeddings

    @property
    def client(self) -> QdrantClient:
        if not self._client:
            with self._lock:
                if not self._client:
                    self._client = self._connect_with_retry()
                    self._ensure_collection()
        return self._client

    @property
    def db(self) -> QdrantVectorStore:
        if not self._db:
            with self._lock:
                if not self._db:
                    self._db = QdrantVectorStore(
                        client=self.client,
                        collection_name=COLLECTION_NAME,
                        embedding=self.embeddings,
                    )
                    logger.info("Qdrant vector store ready.")
        return self._db

    def _connect_with_retry(self) -> QdrantClient:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(f"Connecting to Qdrant [{attempt}/{_MAX_RETRIES}] …")
                client = QdrantClient(url=settings.QDRANT_URL, timeout=10)
                client.get_collections()
                logger.info("Qdrant connected.")
                return client
            except Exception as e:
                if attempt == _MAX_RETRIES:
                    raise CustomException("Qdrant connection failed", e)
                time.sleep(_RETRY_DELAY * attempt)

    def _ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            logger.info(f"Creating collection '{COLLECTION_NAME}' …")
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
        try:
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.chat_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def reload(self):
        with self._lock: self._db = None
        invalidate_search_cache()

    def add(self, chunks: List[Document], chat_id: str):
        if not chunks:
            raise CustomException("No chunks provided.")

        for c in chunks:
            c.metadata = {**(c.metadata or {}), "chat_id": chat_id}

        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # single GPU batch embed
        embeddings = self.embeddings.embed_documents(texts)

        # build points
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"page_content": text, "metadata": meta},
            )
            for text, emb, meta in zip(texts, embeddings, metadatas)
        ]

        # batch upsert — 500 points per call (Qdrant safe limit)
        UPSERT_BATCH = 500
        for i in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i: i + UPSERT_BATCH],
            )

        invalidate_search_cache(chat_id)
        logger.info(f"Indexed {len(chunks)} chunks [chat_id={chat_id}]")

    def delete(self, chat_id: str):
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=_chat_filter(chat_id),
        )
        invalidate_search_cache(chat_id)
        logger.info(f"Deleted vectors [chat_id={chat_id}]")

    def search(self, query: str, k: int = 5,
               chat_id: Optional[str] = None) -> List[Document]:
        key    = _cache_key(query, k, chat_id)
        cached = _get_cached(key)
        if cached is not None: return cached
        kw     = {"filter": _chat_filter(chat_id)} if chat_id else {}
        result = self.db.similarity_search(query, k=k, **kw)
        _set_cache(key, result)
        return result

    def search_with_score(self, query: str, k: int = 5,
                          chat_id: Optional[str] = None) -> List[Tuple[Document, float]]:
        key    = _cache_key(query, k, chat_id) + ":score"
        cached = _get_cached(key)
        if cached is not None: return cached
        kw     = {"filter": _chat_filter(chat_id)} if chat_id else {}
        result = self.db.similarity_search_with_score(query, k=k, **kw)
        _set_cache(key, result)
        return result

    def as_retriever(self, **kw): return self.db.as_retriever(**kw)

    @property
    def vector_count(self) -> int:
        return self.client.get_collection(COLLECTION_NAME).points_count or 0


# ── Singleton ─────────────────────────────────────────────────────────────────
_store = VectorStoreManager()

async def _offload(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_POOL, fn, *args)


# ── Public sync API ───────────────────────────────────────────────────────────
def load_vector_store()          -> QdrantVectorStore:  return _store.db
def load_embeddings()            -> FastGPUEmbeddings:  return _store.embeddings
def get_retriever(**kw):                                return _store.as_retriever(**kw)
def get_vector_count()           -> int:                return _store.vector_count
def refresh_vector_store_cache() -> None:               _store.reload()

def add_documents_to_vector_store(chunks: List[Document], chat_id: str) -> None:
    try:    _store.add(chunks, chat_id)
    except Exception as e: raise CustomException("Vector store add failed", e)

def delete_documents_by_chat(chat_id: str) -> None:
    try:    _store.delete(chat_id)
    except Exception as e: raise CustomException("Vector store delete failed", e)

def similarity_search(query: str, k: int = 5,
                      chat_id: Optional[str] = None) -> List[Document]:
    return _store.search(query, k=k, chat_id=chat_id)

def similarity_search_with_score(query: str, k: int = 5,
                                  chat_id: Optional[str] = None) -> List[Tuple[Document, float]]:
    return _store.search_with_score(query, k=k, chat_id=chat_id)


# ── Public async API ──────────────────────────────────────────────────────────
async def async_add_documents(chunks: List[Document], chat_id: str) -> None:
    await _offload(_store.add, chunks, chat_id)

async def async_delete_by_chat(chat_id: str) -> None:
    await _offload(_store.delete, chat_id)

async def async_search(query: str, k: int = 5,
                       chat_id: Optional[str] = None) -> List[Document]:
    return await _offload(_store.search, query, k, chat_id)

async def async_search_with_score(query: str, k: int = 5,
                                   chat_id: Optional[str] = None) -> List[Tuple[Document, float]]:
    return await _offload(_store.search_with_score, query, k, chat_id)