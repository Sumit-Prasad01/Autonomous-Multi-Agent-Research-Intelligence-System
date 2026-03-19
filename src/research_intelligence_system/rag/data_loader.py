import os
from src.research_intelligence_system.rag.pdf_parser import load_pdf_file, create_text_chunks
from src.research_intelligence_system.rag.vector_store import create_vector_store
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


def process_and_store_pdfs():
    try:

        logger.info("Making the Vector Store.")

        documents = load_pdf_file()

        text_chunks = create_text_chunks(documents)

        create_vector_store(text_chunks)

        logger.info("Vector Store created successfully.")

    
    except Exception as e:
        error_mesasge = CustomException("Failed to load PDFs and create vector store.", e)
        logger.error(str(error_mesasge))
        

if __name__ == "__main__":

    process_and_store_pdfs()