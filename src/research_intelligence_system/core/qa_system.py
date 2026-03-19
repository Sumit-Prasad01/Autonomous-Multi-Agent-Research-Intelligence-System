import asyncio
from typing import Dict

from src.research_intelligence_system.rag.retriever import retrieve_documents
from src.research_intelligence_system.tools.web_search import run_web_search
from src.research_intelligence_system.rag.llm import load_llm
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


# ---------- CONFIG ----------
RAG_TIMEOUT = 8
WEB_TIMEOUT = 10


# ---------- VALIDATION ----------
def is_good_answer(answer: str) -> bool:
    if not answer:
        return False

    answer = answer.lower()

    if "not found" in answer:
        return False

    if len(answer.strip()) < 30:
        return False

    return True


# ---------- ASYNC WRAPPERS ----------
async def run_rag(query: str, chat_id: str, llm_id: str):
    return await asyncio.to_thread(retrieve_documents, query, chat_id, llm_id)


async def run_web(query: str):
    return await asyncio.to_thread(run_web_search, query)


# ---------- MAIN SYSTEM ----------
async def run_qa_system(
    query: str,
    chat_id: str,
    llm_id: str,
    allow_search: bool
) -> Dict:
    """
    Production Agentic RAG:
    - Parallel execution
    - Timeout safe
    - Deterministic decision layer
    """

    try:
        logger.info(f"[QA] chat_id={chat_id} query={query}")

        llm = load_llm(llm_id)

        # ---------- STEP 1: RUN RAG FIRST ----------
        try:
            rag_task = asyncio.wait_for(
                run_rag(query, chat_id, llm_id),
                timeout=RAG_TIMEOUT
            )
            rag_result = await rag_task
        except asyncio.TimeoutError:
            logger.warning("RAG timeout")
            rag_result = ""

        # ---------- STEP 2: DECISION ----------
        if is_good_answer(rag_result):
            return {
                "answer": rag_result.strip(),
                "source": "rag",
                "confidence": 0.9
            }

        logger.info("RAG weak → considering web fallback")

        if not allow_search:
            return {
                "answer": "Answer not found in research papers.",
                "source": "none",
                "confidence": 0.4
            }

        # ---------- STEP 3: WEB SEARCH ----------
        try:
            web_task = asyncio.wait_for(
                run_web(query),
                timeout=WEB_TIMEOUT
            )
            web_context = await web_task
        except asyncio.TimeoutError:
            logger.warning("Web search timeout")
            web_context = ""

        if not web_context:
            return {
                "answer": rag_result or "No relevant information found.",
                "source": "fallback",
                "confidence": 0.5
            }

        # ---------- STEP 4: FINAL SYNTHESIS ----------
        final_prompt = f"""
You are a research assistant.

Combine the following sources to answer the question.

Question:
{query}

RAG Context:
{rag_result}

Web Context:
{web_context}

Rules:
- Prefer research papers over web
- If both useful → combine
- Be concise and factual
"""

        try:
            final_response = await asyncio.to_thread(llm.invoke, final_prompt)
        except Exception as e:
            logger.error("LLM synthesis failed")
            return {
                "answer": rag_result or "Failed to generate answer.",
                "source": "rag_fallback",
                "confidence": 0.5
            }

        if isinstance(final_response, dict):
            final_answer = final_response.get("content", "") or str(final_response)
        elif hasattr(final_response, "content"):
            final_answer = final_response.content
        else:
            final_answer = str(final_response)

        return {
            "answer": final_answer.strip(),
            "source": "hybrid",
            "confidence": 0.85
        }

    except Exception as e:
        logger.exception("QA system failed.")
        raise CustomException("QA system execution failed.", e)