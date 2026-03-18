from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.research_intelligence_system.constants import DB_FAISS_PATH, HF_MODEL_NAME
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)

FAISS_INDEX_PATH = DB_FAISS_PATH

@lru_cache(maxsize=1)
def load_embeddings():
    """
    Load embedding model (cached).
    """

    try:
        logger.info("Loading embedding model...")

        embeddings = HuggingFaceEmbeddings(
            model_name=HF_MODEL_NAME
        )

        return embeddings

    except Exception as e:
        logger.exception("Failed to load embeddings.")
        raise CustomException("Embedding loading failed.", e)


@lru_cache(maxsize=1)
def load_vector_store():
    """
    Load FAISS vector store (cached).
    """

    try:
        logger.info("Loading FAISS vector store...")

        embeddings = load_embeddings()

        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True 
        )

        return db

    except Exception as e:
        logger.exception("Failed to load vector store.")
        raise CustomException("Vector store loading failed.", e)