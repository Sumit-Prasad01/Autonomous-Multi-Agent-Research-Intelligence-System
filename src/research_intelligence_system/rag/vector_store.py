import os 
from langchain_community.vectorstores import FAISS

from src.research_intelligence_system.rag.embedding import get_embedding_model
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.constants import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:

        embedding_model = get_embedding_model()

        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing vector store.")

            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization = True
            )
        
        else:
            logger.warning('No vector store found.')
    
    except Exception as e:
        error_mesasge = CustomException("Failed to load vector store.", e)
        logger.error(str(error_mesasge))


def save_vector_store(text_chunks):
    """Create new vector store if not exists"""

    try:
        
        if not text_chunks:
            raise CustomException("no chunks were found.")

        logger.info("Generating your new vector store")

        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info("Saving vector store.")

        db.save_local(DB_FAISS_PATH)

        return db
    
    except Exception as e:
        error_mesasge = CustomException("Failed to create new vector store", e)
        logger.error(str(error_mesasge))