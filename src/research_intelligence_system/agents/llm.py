from langchain_groq import ChatGroq

from src.research_intelligence_system.config.configuration import settings
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.constants import MODEL

logger = get_logger(__name__)

def load_llm(groq_api_key : str = settings.GROQ_API_KEY, model : str = MODEL):
    try:

        logger.info("Loading LLM from Groq Cloud.")

        llm = ChatGroq(
            model = model,
            api_key = groq_api_key,
            temperature = 0.3,
            max_token = 512
        )

        logger.info("LLM Loaded Successfully.")

        return llm
    
    except Exception as e:
        error_mesasge = CustomException("Failed to load LLM", e)
        logger.error(str(error_mesasge))