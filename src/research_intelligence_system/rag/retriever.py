from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

from src.research_intelligence_system.agents.llm import load_llm
from src.research_intelligence_system.rag.vector_store import load_vector_store
from src.research_intelligence_system.config.configuration import settings
from src.research_intelligence_system.constants import MODEL
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException

logger = get_logger(__name__)


CUSTOM_PROMPT_TEMPLATE = """
You are an AI research assistant.

Answer the question using ONLY the information provided in the context from research papers.

If the context does not contain the answer, say:
"I could not find the answer in the provided research papers."

Provide the answer in 3-5 concise sentences.

Context:
{context}

Question:
{question}

Answer:
"""


def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables = ["context", "question"]
    )


def create_qa_chain():
    try:

        logger.info("Loading vector store for context retrieval.")

        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty.")

        logger.info("Loading LLM.")

        llm = load_llm(
            groq_api_key = settings.GROQ_API_KEY,
            model = MODEL
        )

        if llm is None:
            raise CustomException("Failed to load LLM.")

        logger.info("Creating RetrievalQA chain.")

        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = "stuff",
            retriever = db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents = True,
            chain_type_kwargs = {"prompt": set_custom_prompt()}
        )

        logger.info("QA chain created successfully.")

        return qa_chain

    except Exception as e:
        logger.exception("Failed to create QA chain.")
        raise CustomException("QA chain creation failed.", e)