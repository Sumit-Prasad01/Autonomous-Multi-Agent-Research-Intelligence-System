from src.research_intelligence_system.rag.retriever import create_qa_chain
from src.research_intelligence_system.tools.web_search import run_web_search
from src.research_intelligence_system.rag.llm import load_llm
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


def is_rag_answer_valid(answer: str) -> bool:
    """
    Simple heuristic to check if RAG found a useful answer.
    """

    if not answer:
        return False

    answer_lower = answer.lower()

    if "not found" in answer_lower:
        return False

    if len(answer.strip()) < 20:
        return False

    return True


def run_qa_system(query: str, llm_id: str, allow_search: bool):
    """
    Core QA pipeline:
    1. Try RAG
    2. If weak → optional web search
    3. Final synthesis
    4. Return structured output
    """

    try:
        logger.info("Running QA system...")

        
        rag_chain = create_qa_chain(llm_id)

        rag_response = rag_chain.invoke({"input": query})

        if isinstance(rag_response, dict):
            rag_answer = rag_response.get("output", "") or str(rag_response)
        elif hasattr(rag_response, "content"):
            rag_answer = rag_response.content
        else:
            rag_answer = str(rag_response)

        
        if is_rag_answer_valid(rag_answer):
            return {
                "answer": rag_answer.strip(),
                "source": "rag",
                "confidence": 0.9
            }

        
        if not allow_search:
            return {
                "answer": "Answer not found in research papers.",
                "source": "none",
                "confidence": 0.3
            }

        
        web_context = run_web_search(query)

        if not web_context:
            return {
                "answer": "No relevant information found.",
                "source": "none",
                "confidence": 0.2
            }

        llm = load_llm(llm_id)

        final_prompt = f"""
You are a research assistant.

Use the web information below to answer the question.

Question:
{query}

Web Context:
{web_context}

Provide a concise and accurate answer.
"""

        final_response = llm.invoke(final_prompt)
        
        if isinstance(final_response, dict):
            final_answer = final_response.get("content", "") or str(final_response)
        elif hasattr(final_response, "content"):
            final_answer = final_response.content
        else:
            final_answer = str(final_response)

        return {
            "answer": final_answer.strip(),
            "source": "web",
            "confidence": 0.7
        }

    except Exception as e:
        logger.exception("QA system failed.")
        raise CustomException("QA system execution failed.", e)