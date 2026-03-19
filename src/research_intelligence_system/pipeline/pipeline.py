from src.research_intelligence_system.utils.logger import get_logger
# from src.research_intelligence_system.utils.custom_exception import CustomException
# from src.research_intelligence_system.pipeline.summarization_model_finetuning_pipeline.stage1_data_ingestion_pipeline import DataIngestionTrainingPipeline
# from src.research_intelligence_system.pipeline.summarization_model_finetuning_pipeline.stage2_data_transformation import DataTransformationPieline
from src.research_intelligence_system.pipeline.summarization_model_finetuning_pipeline.stage_3_model_trainer_pipeline import ModelTrainerTrainingPipeline

logger = get_logger(__name__)

# STAGE_NAME = "Data Ingestion Stage"

# try:

#     logger.info(f"Stage {STAGE_NAME} initiated.")

#     data_ingestion_pipeline = DataIngestionTrainingPipeline()
#     data_ingestion_pipeline.initiate_data_ingestion()

#     logger.info(f"Stage {STAGE_NAME} completed.")

# except Exception as e:
#     logger.exception(e)
#     raise CustomException("Failed to download data.", e)


# STAGE_NAME = "Data Transformation Stage"


# try:
#     logger.info(f"Stage {STAGE_NAME} initiated ")

#     data_transformation_pipeline = DataTransformationPieline()
#     data_transformation_pipeline.initiate_data_transformation()

#     logger.info(f"Stage {STAGE_NAME} completed.")

# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Model Trainer Stage"


try:
    logger.info(f"Stage {STAGE_NAME} initiated ")

    model_trainer_pipeline = ModelTrainerTrainingPipeline()
    model_trainer_pipeline.initiate_model_trainer()

    logger.info(f"Stage {STAGE_NAME} completed.")

except Exception as e:
    logger.exception(e)
    raise e