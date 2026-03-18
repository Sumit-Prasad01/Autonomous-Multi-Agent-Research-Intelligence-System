from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.pipeline.summarization_model_finetuning_pipeline.stage1_data_ingestion_pipeline import DataIngestionTrainingPipeline

logger = get_logger(__name__)

STAGE_NAME = "Data Ingestion Stage"

try:

    logger.info(f"Stage {STAGE_NAME} initiated.")

    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()

    logger.info(f"Stage {STAGE_NAME} completed.")

except Exception as e:
    logger.exception(e)
    raise CustomException("Failed to download data.", e)