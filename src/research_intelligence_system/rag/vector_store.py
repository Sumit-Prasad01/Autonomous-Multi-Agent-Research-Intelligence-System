from functools import lru_cache
import os
import threading

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.research_intelligence_system.constants import DB_FAISS_PATH, HF_MODEL_NAME
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)

FAISS_INDEX_PATH = DB_FAISS_PATH

# Thread lock for safe writes
faiss_lock = threading.Lock()


# ---------- EMBEDDINGS ----------
@lru_cache(maxsize=1)
def load_embeddings():
    try:
        logger.info("Loading embedding model (cached)")

        return HuggingFaceEmbeddings(
            model_name=HF_MODEL_NAME,
            model_kwargs={
                "local_files_only": True
            },
            encode_kwargs={
                "normalize_embeddings": True  #improves similarity search
            }
        )

    except Exception as e:
        logger.exception("Embedding load failed")
        raise CustomException("Embedding model loading failed", e)


# ---------- VECTOR STORE ----------
@lru_cache(maxsize=1)
def load_vector_store():
    try:
        embeddings = load_embeddings()

        if os.path.exists(FAISS_INDEX_PATH):
            logger.info("Loading existing FAISS index")

            db = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.warning("No FAISS index found → initializing empty store")

            db = FAISS.from_texts(
                ["initial placeholder"],
                embedding=embeddings
            )

        return db

    except Exception as e:
        logger.exception("Failed to load vector store.")
        raise CustomException("Vector store loading failed.", e)


# ---------- ADD DOCUMENTS ----------
def add_documents_to_vector_store(text_chunks, chat_id):
    """
    🚀 Production-safe document ingestion:
    - Thread-safe writes
    - Metadata enforced
    - Persistent storage
    """

    try:
        if not text_chunks:
            raise CustomException("No chunks provided for indexing.")

        logger.info(f"[VECTOR STORE] Adding docs for chat_id={chat_id}")

        db = load_vector_store()

        for chunk in text_chunks:
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}

            chunk.metadata["chat_id"] = chat_id

        # thread-safe FAISS update
        with faiss_lock:
            db.add_documents(text_chunks)
            db.save_local(FAISS_INDEX_PATH)

        logger.info("Documents successfully added and saved.")

    except Exception as e:
        logger.exception("Failed to add documents.")
        raise CustomException("Failed to update vector store", e)


# ---------- DELETE BY CHAT ----------
def delete_documents_by_chat(chat_id: str):
    """
    Future-ready cleanup (important for production)
    """

    try:
        logger.info(f"[VECTOR STORE] Deleting docs for chat_id={chat_id}")

        db = load_vector_store()

        with faiss_lock:
            # FAISS doesn't support direct delete → rebuild index
            all_docs = list(db.docstore._dict.values())

            remaining_docs = [
                doc for doc in all_docs
                if doc.metadata.get("chat_id") != chat_id
            ]

            embeddings = load_embeddings()

            new_db = FAISS.from_documents(remaining_docs, embeddings)
            new_db.save_local(FAISS_INDEX_PATH)

            # refresh cache
            load_vector_store.cache_clear()

        logger.info("Documents deleted successfully.")

    except Exception as e:
        logger.exception("Failed to delete documents.")
        raise CustomException("Failed to delete documents", e)


# ---------- CACHE REFRESH ----------
def refresh_vector_store_cache():
    """
    Reload FAISS after updates
    """
    logger.info("Refreshing FAISS cache")
    load_vector_store.cache_clear()