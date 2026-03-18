from functools import lru_cache
from langchain_core.tools import tool

from src.research_intelligence_system.rag.retriever import create_qa_chain
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chain():
    return create_qa_chain()


@tool
def research_paper_qa(query: str) -> str:
    """
    Answer questions strictly from research papers in the vector database.

    Use ONLY when the query requires academic/research-based answers.
    Do NOT use for general knowledge or web queries.

    Input: Natural language question
    Output: Concise answer from research context only
    """

    try:
        qa_chain = get_chain()

        response = qa_chain.invoke({"input": query})

        answer = response.get("answer", "").strip()

        if not answer:
            return "No answer found in research papers."

        return answer

    except Exception as e:
        logger.exception("RAG tool execution failed.")
        raise CustomException("RAG tool failed.", e)