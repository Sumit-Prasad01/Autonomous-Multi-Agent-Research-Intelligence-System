from functools import lru_cache

from langchain_groq import ChatGroq

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


@lru_cache(maxsize=5)
def load_llm(model: str):
    """
    Load and cache Groq LLM.

    Args:
        model (str): Model name (passed from user)

    Returns:
        ChatGroq instance
    """

    try:
        logger.info(f"Loading LLM: {model}")

        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=model,
            streaming=True,        
            temperature=0.2        
        )

        return llm

    except Exception as e:
        logger.exception("Failed to load LLM.")
        raise CustomException("LLM loading failed.", e)