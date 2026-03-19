from functools import lru_cache
from typing import List
import re

from langchain_community.tools.tavily_search import TavilySearchResults

from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


# ---------- CACHE TOOL ----------
@lru_cache(maxsize=1)
def get_search_tool():
    try:
        logger.info("Initializing Tavily search tool...")

        return TavilySearchResults(
            max_results=settings.TAVILY_MAX_RESULTS
        )

    except Exception as e:
        logger.exception("Failed to initialize Tavily.")
        raise CustomException("Tavily initialization failed.", e)


# ---------- QUERY CLEANING ----------
def optimize_query(query: str) -> str:
    """
    Improve query for better Tavily results
    """
    query = query.strip()

    # remove unnecessary punctuation
    query = re.sub(r"[^\w\s\-]", " ", query)

    # compress spaces
    query = re.sub(r"\s+", " ", query)

    # add research context boost
    return f"{query} research paper explanation"


# ---------- RESULT CLEANING ----------
def clean_results(results: List[dict]) -> str:
    cleaned = []

    for r in results:
        content = r.get("content", "")
        if not content:
            continue

        text = content.strip()

        # remove very short/noisy results
        if len(text) < 50:
            continue

        cleaned.append(text)

    return "\n\n".join(cleaned[: settings.TAVILY_MAX_RESULTS])


# ---------- MAIN FUNCTION ----------
def run_web_search(query: str) -> str:
    """
    🚀 Production Web Search:
    - Query optimization
    - Noise filtering
    - Safe fallback
    """

    try:
        logger.info(f"[WEB SEARCH] query={query}")

        tool = get_search_tool()

        optimized_query = optimize_query(query)

        results = tool.invoke({"query": optimized_query})

        if not results:
            return ""

        cleaned_context = clean_results(results)

        return cleaned_context or ""

    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return ""