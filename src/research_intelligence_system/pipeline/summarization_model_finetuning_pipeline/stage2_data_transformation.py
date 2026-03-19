from src.research_intelligence_system.config.configuration import ConfigurationManager
from src.research_intelligence_system.components.data_transformation import DataTransformation
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformationPieline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config = data_transformation_config)
        data_transformation.transform()