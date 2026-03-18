import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.constants import PDF_PATH, CHUNK_OVERLAP, CHUNK_SIZE


logger = get_logger(__name__)

def load_pdf_file():
    try:

        if not os.path.exists(PDF_PATH):
            raise CustomException("Data path does not exists.")
        
        logger.info(f"Loading files from {PDF_PATH}")

        loader = DirectoryLoader(PDF_PATH, glob = "*.pdf", loader_cls = PyPDFLoader)

        documents = loader.load()

        if not documents:
            logger.warning("No PDFs were found.")
        else:
            logger.info(f"Successfully fetched {len(documents)} documents.")

        return documents
    
    except Exception as e:
        error_message = CustomException("Failed o load PDFs.", e)
        logger.error(str(error_message))
        return []
    

def create_text_chunks(documents):
    try:

        if not documents:
            raise CustomException("No documents were found.")
        
        logger.info(f"Splitting {len(documents)} documents into chunks.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)} text chunks.")

        return text_chunks

    except Exception as e:
        error_mesasge = CustomException("Failed to generate text chunks.", e)
        logger.error(str(error_mesasge))
        return []