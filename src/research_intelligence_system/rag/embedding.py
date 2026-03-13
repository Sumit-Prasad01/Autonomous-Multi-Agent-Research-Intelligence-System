from langchain_huggingface import HuggingFaceEmbeddings

from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.constants import HF_MODEL_NAME

logger = get_logger(__name__)


def get_embedding_model():
    try:

        logger.info("Initializing our HuggingFace embedding model.")

        model = HuggingFaceEmbeddings(model_name = HF_MODEL_NAME)

        logger.info("HuggingFace embedding model loaded successfully.")

        return model
    
    except Exception as e:
        error_mesasge = CustomException("Error occured while loading embedding model.", e)
        logger.error(str(error_mesasge))
        raise error_mesasge