from src.research_intelligence_system.core.qa_system import run_qa_system
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


def process_chat(request):
    """
    Entry point for chat requests.
    Extracts query and calls QA system.
    """

    try:
        if not request.messages:
            return {
                "answer": "No input provided.",
                "source": "none",
                "confidence": 0.0
            }

        latest_query = request.messages[-1].content

        response = run_qa_system(
            query=latest_query,
            llm_id=request.llm_id,
            allow_search=request.allow_search
        )

        return response

    except Exception as e:
        logger.exception("Chat service failed.")
        return {
            "answer": "Something went wrong while processing your request.",
            "source": "error",
            "confidence": 0.0
        }