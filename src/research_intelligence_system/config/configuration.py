from src.research_intelligence_system.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.research_intelligence_system.utils.common import read_yaml, create_directories
from src.research_intelligence_system.entity import DataIngestionConfig


class ConfigurationManager:

    def __init__(self, config_file_path = CONFIG_FILE_PATH, params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            dataset_name = config.dataset_name,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config