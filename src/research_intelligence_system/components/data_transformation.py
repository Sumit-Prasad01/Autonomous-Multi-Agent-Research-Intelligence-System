import os
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from src.research_intelligence_system.entity import DataTransformationConfig
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            cache_dir="./hf_cache",
            local_files_only=False
        )

    def clean_text(self, text):
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r'@xcite', '', text)
        text = re.sub(r'@xmath\d+', '', text)
        text = text.replace("\n", " ")

        text = re.sub(r'\s+,', ',', text)
        text = re.sub(r'\s+\.', '.', text)
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip().lower()


    def load_data(self):
        dataset = load_dataset("csv", data_files=self.config.data_path)
        return dataset


    def apply_cleaning(self, dataset):

        def clean_batch(example):
            return {
                "input_text": self.clean_text(example.get("text")),
                "target_text": self.clean_text(example.get("summary"))
            }

        dataset = dataset.map(clean_batch)

        cols_to_remove = [col for col in ["text", "summary"] if col in dataset["train"].column_names]
        dataset = dataset.remove_columns(cols_to_remove)

        return dataset


    def filter_dataset(self, dataset):

        def filter_empty(example):
            return (
                example["input_text"] and example["input_text"].strip() != "" and
                example["target_text"] and example["target_text"].strip() != ""
            )

        return dataset.filter(filter_empty)


    def validate_dataset(self, dataset):

        stats = {
            "total": 0,
            "valid": 0,
            "removed_empty": 0,
            "removed_short": 0,
            "avg_input_length": 0
        }

        lengths = []

        def validate(example):
            stats["total"] += 1

            input_text = example["input_text"]
            target_text = example["target_text"]

            if not input_text or not target_text:
                stats["removed_empty"] += 1
                return False

            if len(input_text) < 30:
                stats["removed_short"] += 1
                return False

            lengths.append(len(input_text))
            stats["valid"] += 1
            return True

        dataset = dataset.filter(validate)   

        if lengths:
            stats["avg_input_length"] = sum(lengths) / len(lengths)

        return dataset, stats

    def log_stats(self, stats):

        logger.info("\n📊 DATASET QUALITY REPORT")
        logger.info(f"Total samples: {stats['total']}")
        logger.info(f"Valid samples: {stats['valid']}")
        logger.info(f"Removed (empty): {stats['removed_empty']}")
        logger.info(f"Removed (too short): {stats['removed_short']}")
        logger.info(f"Avg input length: {stats['avg_input_length']:.2f}")


    def tokenize_function(self, examples):

        inputs = ["summarize: " + text for text in examples["input_text"]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,   
            truncation=True,
            padding=False   
        )

        labels = self.tokenizer(
            examples["target_text"],
            max_length=self.config.max_target_length,  
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    def transform(self):

        os.makedirs(self.config.root_dir, exist_ok=True)

        logger.info("Loading dataset...")
        dataset = self.load_data()

        logger.info("Cleaning dataset...")
        dataset = self.apply_cleaning(dataset)

        logger.info("Validating dataset...")
        dataset, stats = self.validate_dataset(dataset)
        self.log_stats(stats)

        logger.info("Filtering empty rows...")
        dataset = self.filter_dataset(dataset)

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["input_text", "target_text"]
        )

        save_path = os.path.join(self.config.root_dir, "tokenized_dataset")
        tokenized_dataset.save_to_disk(save_path)

        logger.info(f"Tokenized dataset saved at: {save_path}")

        return tokenized_dataset