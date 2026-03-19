from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor

from src.research_intelligence_system.rag.llm import load_llm
from src.research_intelligence_system.rag.vector_store import load_vector_store
from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.rag.reranker import rerank_documents

logger = get_logger(__name__)


# ---------- CACHE VECTOR DB ----------
@lru_cache(maxsize=1)
def get_vector_db():
    return load_vector_store()


# ---------- BUILD BM25 ----------
@lru_cache(maxsize=1)
def get_bm25_retriever():
    db = get_vector_db()

    docs = db.docstore._dict.values()  # raw docs from FAISS

    bm25 = BM25Retriever.from_documents(list(docs))
    bm25.k = settings.RETRIEVAL_K

    return bm25


# ---------- FORMAT ----------
def format_docs(docs, query: str) -> str:
    if not docs:
        return ""

    texts = []
    for doc in docs:
        content = doc.page_content.strip()
        if content:
            texts.append(content)

    #APPLY RERANKING
    ranked_texts = rerank_documents(
        query=query,
        docs=texts,
        top_k=5
    )

    return "\n\n".join(ranked_texts)


# ---------- HYBRID RETRIEVER ----------
def get_hybrid_retriever(chat_id: str, llm):
    db = get_vector_db()

    dense_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.RETRIEVAL_K,
            "lambda_mult": settings.RETRIEVAL_LAMBDA,
            "filter": {"chat_id": chat_id}
        }
    )

    bm25 = get_bm25_retriever()

    # Ensemble (Hybrid)
    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, bm25],
        weights=[0.6, 0.4]  # tuneable
    )

    compressor = LLMChainExtractor.from_llm(llm)

    return ContextualCompressionRetriever(
        base_retriever=hybrid,
        base_compressor=compressor
    )


# ---------- MAIN ----------
def retrieve_documents(query: str, chat_id: str, llm_id: str) -> str:
    try:
        logger.info(f"[HYBRID RETRIEVER + RERANK] query={query}")

        llm = load_llm(llm_id)

        retriever = get_hybrid_retriever(chat_id, llm)

        docs = retriever.invoke(query)

        context = format_docs(docs, query)

        return context or ""

    except Exception as e:
        logger.error(f"Retriever failed: {str(e)}")
        return ""