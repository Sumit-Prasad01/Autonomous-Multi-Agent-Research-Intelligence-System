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
    logger.info("Loading embedding model (ONLY ONCE)")
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"local_files_only": True}  # prevents HF calls
    )


#Cache vector store
@lru_cache(maxsize=1)
def load_vector_store():
    """
    Load FAISS vector store (cached).
    """

    try:
        logger.info("🔥 Loading FAISS vector store (ONLY ONCE)")

        embeddings = load_embeddings()  #reuse cached embeddings

        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return db

    except Exception as e:
        logger.exception("Failed to load vector store.")
        raise CustomException("Vector store loading failed.", e)
    

def create_vector_store(text_chunks):
    """Create new vector store if not exists."""

    try:

        if not text_chunks:
            raise CustomException("No chunks were found.")
        
        logger.info("Generating your new vector store")

        embedding_model = HuggingFaceEmbeddings(
            model_name=HF_MODEL_NAME
        )

        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info("Saving vector store.")

        db.save_local(DB_FAISS_PATH)

        logger.info("Vector store saved successfully.")

        return db
    
    except Exception as e:
        error_mesasge = CustomException("Failed to create new vector store", e)
        logger.error(str(error_mesasge))