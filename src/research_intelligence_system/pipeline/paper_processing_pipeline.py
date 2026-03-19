import asyncio
from typing import List

from src.research_intelligence_system.rag.vector_store import (
    add_documents_to_vector_store,
    refresh_vector_store_cache
)
from src.research_intelligence_system.rag.pdf_parser import (
    load_pdf_file,
    create_text_chunks
)
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


# ---------- CONFIG ----------
BATCH_SIZE = 64   # controls memory + speed


# ---------- ASYNC WRAPPERS ----------
async def load_pdf_async(pdf_path: str):
    return await asyncio.to_thread(load_pdf_file, pdf_path)


async def chunk_docs_async(documents):
    return await asyncio.to_thread(create_text_chunks, documents)


async def store_batch_async(batch, chat_id: str):
    return await asyncio.to_thread(add_documents_to_vector_store, batch, chat_id)


# ---------- BATCHING ----------
def batch_chunks(chunks: List, batch_size: int):
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]


# ---------- MAIN PIPELINE ----------
async def ingest_pdf(chat_id: str, pdf_path: str):
    """
    🚀 Production Ingestion Pipeline:
    - Async execution
    - Batch processing
    - Metadata enforced
    - Scalable for large PDFs
    """

    try:
        logger.info(f"[INGESTION] chat_id={chat_id}")
        logger.info(f"[PDF] path={pdf_path}")

        # ---------- LOAD ----------
        documents = await load_pdf_async(pdf_path)

        if not documents:
            raise CustomException("No documents loaded from PDF.")

        logger.info(f"Loaded {len(documents)} pages")

        # ---------- CHUNK ----------
        chunks = await chunk_docs_async(documents)

        if not chunks:
            raise CustomException("Chunking failed.")

        logger.info(f"Generated {len(chunks)} chunks")

        # ---------- METADATA ----------
        for chunk in chunks:
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}

            chunk.metadata["chat_id"] = chat_id

        # ---------- BATCH STORE ----------
        tasks = []

        for batch in batch_chunks(chunks, BATCH_SIZE):
            tasks.append(store_batch_async(batch, chat_id))

        await asyncio.gather(*tasks)

        # ---------- REFRESH CACHE ----------
        await asyncio.to_thread(refresh_vector_store_cache)

        logger.info("Ingestion completed successfully.")

        return True

    except Exception as e:
        logger.exception("Ingestion pipeline failed.")
        raise CustomException("PDF ingestion failed.", e)