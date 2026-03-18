from functools import lru_cache

from langchain_community.tools.tavily_search import TavilySearchResults

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_search_tool():
    try:
        logger.info("Initializing Tavily search tool...")

        tool = TavilySearchResults(
            max_results=settings.TAVILY_MAX_RESULTS
        )

        return tool

    except Exception as e:
        logger.exception("Failed to initialize Tavily.")
        raise CustomException("Tavily initialization failed.", e)


def run_web_search(query: str) -> str:
    """
    Execute web search and return cleaned context string.
    """

    try:
        tool = get_search_tool()

        results = tool.invoke({"query": query})

        if not results:
            return ""

        cleaned_results = []

        for r in results:
            content = r.get("content", "")
            if content:
                cleaned_results.append(content.strip())

        return "\n\n".join(cleaned_results[:settings.TAVILY_MAX_RESULTS])

    except Exception as e:
        logger.exception("Web search failed.")
        raise CustomException("Web search failed.", e)