import re
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer
from src.research_intelligence_system.utils.logger import get_logger
from src.research_intelligence_system.config.configuration import DataTransformationConfig


logger = get_logger(__name__)


class DataTransformation:

    def __init__(self, config=DataTransformationConfig):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_name)

        self.MAX_INPUT_TOKENS = 512
        self.CHUNK_SIZE = 350
        self.OVERLAP = 80



    def clean_text(self, text: str) -> str:
        text = re.sub(r'@xcite', '', text)
        text = re.sub(r'@xmath\d+', '', text)

        text = text.replace("\n", " ")

        text = re.sub(r'\s+,', ',', text)
        text = re.sub(r'\s+\.', '.', text)
        text = re.sub(r'\s*-\s*', '-', text)

        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()



    def chunk_text(self, text):
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False
        )["input_ids"]

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.CHUNK_SIZE
            chunk_tokens = tokens[start:end]

            if len(chunk_tokens) == 0:
                break

            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )

            chunks.append(chunk_text)

            start += self.CHUNK_SIZE - self.OVERLAP

        return chunks


    def preprocess(self, input_path, output_path):
        df = pd.read_csv(input_path)

        inputs = []
        targets = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            raw_text = str(row["text"])
            raw_summary = str(row["summary"])

            clean_txt = self.clean_text(raw_text)
            clean_sum = self.clean_text(raw_summary)

            if len(clean_txt) < 50:
                continue

            chunks = self.chunk_text(clean_txt)

            for chunk in chunks:
                inputs.append(f"summarize: {chunk}")
                targets.append(clean_sum)

        final_df = pd.DataFrame({
            "input_text": inputs,
            "target_text": targets
        })

        final_df.to_csv(output_path, index=False)

        logger.info(f"CSV Ready! Size: {len(final_df)}")



    def tokenize_dataset(self, csv_path, save_path):

        dataset = load_dataset("csv", data_files=csv_path)

        def tokenize_function(example):
            model_inputs = self.tokenizer(
                example["input_text"],
                max_length=512,
                truncation=True,        
                padding="max_length"
            )

            labels = self.tokenizer(
                example["target_text"],
                max_length=128,
                truncation=True,
                padding="max_length"
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["input_text", "target_text"]
        )

        tokenized_dataset.save_to_disk(save_path)

        logger.info("Tokenized dataset saved successfully!")


    def transform(self):
        os.makedirs(self.config.root_dir, exist_ok=True)

        csv_path = os.path.join(self.config.root_dir, "cleaned.csv")
        tokenized_path = os.path.join(self.config.root_dir, "tokenized")

        self.preprocess(self.config.data_path, csv_path)

        self.tokenize_dataset(csv_path, tokenized_path)