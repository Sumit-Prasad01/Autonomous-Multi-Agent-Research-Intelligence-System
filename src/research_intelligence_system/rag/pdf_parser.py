import os
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.constants import CHUNK_OVERLAP, CHUNK_SIZE
from src.research_intelligence_system.agents.parsing_agent import ParsingAgent

logger = get_logger(__name__)


# ---------- CLEAN TEXT ----------
def clean_text(text: str) -> str:
    """
    Remove noise from PDF text
    """
    if not text:
        return ""

    # remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # remove weird characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # remove references section (optional improvement)
    text = re.sub(r"References.*", "", text, flags=re.IGNORECASE)

    return text.strip()


# ---------- LOAD PDF ----------
def load_pdf_file(pdf_path: str) -> List[Document]:
    try:
        if not os.path.exists(pdf_path):
            raise CustomException(f"File does not exist: {pdf_path}")

        logger.info(f"[PDF LOAD] path={pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise CustomException("No content extracted from PDF.")

        # clean page content
        cleaned_docs = []
        for doc in documents:
            cleaned_text = clean_text(doc.page_content)

            if len(cleaned_text) < 30:
                continue

            cleaned_docs.append(
                Document(
                    page_content=cleaned_text,
                    metadata=doc.metadata
                )
            )

        logger.info(f"Loaded {len(cleaned_docs)} cleaned pages")

        return cleaned_docs

    except Exception as e:
        logger.exception("PDF loading failed")
        return []


# ---------- CHUNKING ----------
def create_text_chunks(documents: List[Document]) -> List[Document]:
    try:
        if not documents:
            raise CustomException("No documents were found.")

        logger.info(f"[PARSING] documents={len(documents)}")

        # ---------- STRUCTURED PARSING ----------
        parser = ParsingAgent()
        parsed_documents = parser.parse_documents(documents)

        logger.info("Parsing completed → chunking started")

        # ---------- SMART SPLITTING ----------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n", "\n", ".", " ", ""   #better semantic splitting
            ]
        )

        chunks = text_splitter.split_documents(parsed_documents)

        # ---------- CLEAN CHUNKS ----------
        cleaned_chunks = []
        for chunk in chunks:
            text = chunk.page_content.strip()

            if len(text) < 50:
                continue

            cleaned_chunks.append(
                Document(
                    page_content=text,
                    metadata=chunk.metadata
                )
            )

        logger.info(f"Generated {len(cleaned_chunks)} clean chunks")

        return cleaned_chunks

    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        return []