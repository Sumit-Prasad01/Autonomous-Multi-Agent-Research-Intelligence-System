import os
import pandas as pd
from datasets import load_dataset

from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.utils.custom_exception import CustomException
from src.research_intelligence_system.config.config import settings

logger = get_logger(__name__)


class DataIngestion:

    def __init__(self, config):
        self.config = config
        self.dataset = None

    def download_file(self):
        """
        Downloads dataset from HuggingFace using environment token
        """

        try:
            logger.info("Downloading dataset from HuggingFace")

            hf_token = settings.HF_TOKEN

            if hf_token is None:
                raise CustomException("HF_TOKEN environment variable not found")

            # Download only subset for faster ingestion
            train_dataset = load_dataset(
                self.config.dataset_name,
                split="train[:15000]",
                token=hf_token
            )

            val_dataset = load_dataset(
                self.config.dataset_name,
                split="validation[:2000]",
                token=hf_token
            )

            test_dataset = load_dataset(
                self.config.dataset_name,
                split="test[:2000]",
                token=hf_token
            )

            self.dataset = {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            }

            logger.info("Dataset downloaded successfully")

        except Exception as e:
            raise CustomException("Failed to download dataset", e)


    def save_dataset(self):
        """
        Saves dataset into artifacts directory as CSV
        """

        try:

            if self.dataset is None:
                raise CustomException("Dataset not loaded. Run download_file() first.")

            save_path = self.config.unzip_dir
            os.makedirs(save_path, exist_ok=True)

            logger.info("Saving dataset to artifacts directory")

            train_df = pd.DataFrame(self.dataset["train"])
            val_df = pd.DataFrame(self.dataset["validation"])
            test_df = pd.DataFrame(self.dataset["test"])

            # Keep only required columns
            train_df = train_df[["article", "abstract"]]
            val_df = val_df[["article", "abstract"]]
            test_df = test_df[["article", "abstract"]]

            # Rename columns
            train_df.columns = ["text", "summary"]
            val_df.columns = ["text", "summary"]
            test_df.columns = ["text", "summary"]

            train_df.to_csv(os.path.join(save_path, "train.csv"), index=False)
            val_df.to_csv(os.path.join(save_path, "validation.csv"), index=False)
            test_df.to_csv(os.path.join(save_path, "test.csv"), index=False)

            logger.info("Dataset saved successfully")

        except Exception as e:
            raise CustomException("Failed to save dataset", e)