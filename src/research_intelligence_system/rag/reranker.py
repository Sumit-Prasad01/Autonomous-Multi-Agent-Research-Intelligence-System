from functools import lru_cache
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


# ---------- LOAD MODEL ----------
@lru_cache(maxsize=1)
def load_reranker():
    logger.info("Loading reranker model...")
    return CrossEncoder("BAAI/bge-reranker-base")


# ---------- RERANK FUNCTION ----------
def rerank_documents(query: str, docs: List[str], top_k: int = 5) -> List[str]:
    """
    Rerank retrieved documents using cross-encoder
    """

    try:
        if not docs:
            return []

        model = load_reranker()

        pairs: List[Tuple[str, str]] = [(query, doc) for doc in docs]

        scores = model.predict(pairs)

        scored_docs = list(zip(docs, scores))

        # sort by score descending
        ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_k]]

    except Exception as e:
        logger.error(f"Reranker failed: {str(e)}")
        return docs[:top_k]