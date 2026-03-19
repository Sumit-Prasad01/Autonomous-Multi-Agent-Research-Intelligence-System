from src.research_intelligence_system.config.configuration import ConfigurationManager
from src.research_intelligence_system.components.data_ingestion import DataIngestion
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

class DataIngestionTrainingPipeline:

    def __init__(self):
        pass

    def initiate_data_ingestion(self):

        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)

        data_ingestion.download_file()
        data_ingestion.save_dataset()
