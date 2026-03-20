from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import faiss
import numpy as np
from langchain_classic.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.research_intelligence_system.constants import DB_FAISS_PATH, HF_MODEL_NAME
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE      = 64
IVF_THRESHOLD         = 10_000
IVF_NLIST, IVF_NPROBE = 128, 16
_POOL                 = ThreadPoolExecutor(max_workers=4, thread_name_prefix="faiss")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():         return "cuda"
        if torch.backends.mps.is_available(): return "mps"
    except ImportError:
        pass
    return "cpu"


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": _detect_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH_SIZE},
    )


def _upgrade_to_ivf(db: FAISS) -> None:
    """Upgrade flat index → IVFFlat once collection crosses threshold."""
    n = db.index.ntotal
    if isinstance(db.index, faiss.IndexIVFFlat):
        db.index.nprobe = IVF_NPROBE
        return
    if n < IVF_THRESHOLD:
        return

    logger.info(f"Upgrading to IVFFlat (n={n}) …")
    d   = db.index.d
    idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, IVF_NLIST, faiss.METRIC_L2)
    idx.nprobe = IVF_NPROBE
    vecs = np.zeros((n, d), dtype="float32")
    db.index.reconstruct_n(0, n, vecs)
    idx.train(vecs); idx.add(vecs)
    db.index = idx
    logger.info("IVFFlat upgrade done.")


# ── Store Manager (Singleton + RW-lock) ──────────────────────────────────────
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
        self._db: Optional[FAISS]                    = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._wlock   = threading.Lock()
        self._rlock   = threading.Lock()
        self._rcnt    = 0
        self._no_rdr  = threading.Event()
        self._no_rdr.set()

    # ── RW guards ────────────────────────────────────────────────────────────
    def _r_acquire(self):
        with self._rlock:
            self._rcnt += 1
            self._no_rdr.clear()

    def _r_release(self):
        with self._rlock:
            self._rcnt -= 1
            if not self._rcnt: self._no_rdr.set()

    def _w_acquire(self): self._no_rdr.wait(); self._wlock.acquire()
    def _w_release(self): self._wlock.release()

    # ── Lazy properties ───────────────────────────────────────────────────────
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if not self._embeddings:
            self._embeddings = _build_embeddings()
        return self._embeddings

    @property
    def db(self) -> FAISS:
        if not self._db:
            self._w_acquire()
            try:
                if not self._db:
                    self._db = (
                        FAISS.load_local(DB_FAISS_PATH, self.embeddings,
                                         allow_dangerous_deserialization=True)
                        if os.path.exists(DB_FAISS_PATH)
                        else FAISS.from_texts(["__init__"], self.embeddings)
                    )
                    logger.info(f"FAISS ready — {self._db.index.ntotal} vectors")
            finally:
                self._w_release()
        return self._db

    def reload(self):
        self._w_acquire()
        try:    self._db = None
        finally: self._w_release()

    # ── Writes ────────────────────────────────────────────────────────────────
    def add(self, chunks: List[Document], chat_id: str):
        if not chunks: raise CustomException("No chunks provided.")
        for c in chunks:
            c.metadata = {**(c.metadata or {}), "chat_id": chat_id}
        self._w_acquire()
        try:
            for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                self.db.add_documents(chunks[i: i + EMBED_BATCH_SIZE])
            _upgrade_to_ivf(self.db)
            self.db.save_local(DB_FAISS_PATH)
            logger.info(f"Indexed {len(chunks)} chunks [chat_id={chat_id}]")
        finally:
            self._w_release()

    def delete(self, chat_id: str):
        self._w_acquire()
        try:
            ids = [k for k, v in self.db.docstore._dict.items()
                   if v.metadata.get("chat_id") == chat_id]
            if not ids:
                logger.warning(f"No docs for chat_id={chat_id}"); return
            if hasattr(self.db, "delete"):
                self.db.delete(ids)
            else:
                remaining = [v for v in self.db.docstore._dict.values()
                             if v.metadata.get("chat_id") != chat_id]
                self._db = FAISS.from_documents(remaining, self.embeddings)
            self.db.save_local(DB_FAISS_PATH)
            logger.info(f"Deleted {len(ids)} vectors [chat_id={chat_id}]")
        finally:
            self._w_release()

    # ── Reads ─────────────────────────────────────────────────────────────────
    def search(self, query: str, k=5, chat_id=None) -> List[Document]:
        self._r_acquire()
        try:
            kw = {"filter": {"chat_id": chat_id}} if chat_id else {}
            return self.db.similarity_search(query, k=k, **kw)
        finally: self._r_release()

    def search_with_score(self, query: str, k=5, chat_id=None) -> List[tuple]:
        self._r_acquire()
        try:
            kw = {"filter": {"chat_id": chat_id}} if chat_id else {}
            return self.db.similarity_search_with_score(query, k=k, **kw)
        finally: self._r_release()

    def as_retriever(self, **kw): return self.db.as_retriever(**kw)

    @property
    def vector_count(self) -> int: return self.db.index.ntotal


# ── Module-level singleton & thread pool ─────────────────────────────────────
_store = VectorStoreManager()

async def _offload(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(_POOL, fn, *args)


# ── Public sync API (drop-in compatible) ─────────────────────────────────────
def load_vector_store()          -> FAISS:                return _store.db
def load_embeddings()            -> HuggingFaceEmbeddings: return _store.embeddings
def get_retriever(**kw):                                   return _store.as_retriever(**kw)
def get_vector_count()           -> int:                   return _store.vector_count
def refresh_vector_store_cache() -> None:                  _store.reload()

def add_documents_to_vector_store(chunks: List[Document], chat_id: str) -> None:
    try:    _store.add(chunks, chat_id)
    except Exception as e: raise CustomException("Vector store add failed", e)

def delete_documents_by_chat(chat_id: str) -> None:
    try:    _store.delete(chat_id)
    except Exception as e: raise CustomException("Vector store delete failed", e)

def similarity_search(query: str, k=5, chat_id=None) -> List[Document]:
    return _store.search(query, k=k, chat_id=chat_id)

def similarity_search_with_score(query: str, k=5, chat_id=None) -> List[tuple]:
    return _store.search_with_score(query, k=k, chat_id=chat_id)


# ── Public async API ──────────────────────────────────────────────────────────
async def async_add_documents(chunks: List[Document], chat_id: str) -> None:
    await _offload(_store.add, chunks, chat_id)

async def async_delete_by_chat(chat_id: str) -> None:
    await _offload(_store.delete, chat_id)

async def async_search(query: str, k=5, chat_id=None) -> List[Document]:
    return await _offload(_store.search, query, k, chat_id)

async def async_search_with_score(query: str, k=5, chat_id=None) -> List[tuple]:
    return await _offload(_store.search_with_score, query, k, chat_id)
