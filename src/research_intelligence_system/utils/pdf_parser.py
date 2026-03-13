import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 

from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.constants import PDF_PATH

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
    

    