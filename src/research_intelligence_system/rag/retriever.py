# src/research_intelligence_system/rag/retriever.py

from functools import lru_cache

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor

from src.research_intelligence_system.rag.llm import load_llm
from src.research_intelligence_system.rag.vector_store import load_vector_store
from src.research_intelligence_system.config.settings import settings
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


CUSTOM_PROMPT_TEMPLATE = """
You are a research assistant.

Answer ONLY from the given context.
If not found, say: "Answer not found in provided research papers."

Answer in {max_sentences} concise sentences.

Context:
{context}

Question:
{input}
"""


def get_prompt():
    return PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@lru_cache(maxsize=5)
def create_qa_chain(llm_id: str):
    """
    Build LCEL RAG pipeline:
    - MMR retrieval
    - Context compression
    - Prompt → LLM
    """

    try:
        logger.info(f"Creating RAG chain for model: {llm_id}")

        db = load_vector_store()
        llm = load_llm(llm_id)

        base_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.RETRIEVAL_K,
                "lambda_mult": settings.RETRIEVAL_LAMBDA
            }
        )

        compressor = LLMChainExtractor.from_llm(llm)

        retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )

        prompt = get_prompt()

        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["input"]) 
                        | retriever 
                        | RunnableLambda(format_docs),

                "input": RunnableLambda(lambda x: x["input"]),

                "max_sentences": RunnableLambda(lambda _: settings.MAX_ANSWER_SENTENCES)
            }
            | prompt
            | llm
        )

        return rag_chain

    except Exception as e:
        logger.exception("Failed to create RAG chain.")
        raise CustomException("RAG pipeline creation failed.", e)