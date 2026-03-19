import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

from src.research_intelligence_system.config.configuration import ModelTrainerConfig
from src.research_intelligence_system.utils.logger import get_logger


logger = get_logger(__name__)


class TrainerPipeline:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        self.dataset = load_from_disk(self.config.data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)

        # LoRA Config (T5-specific)
        lora_config = LoraConfig(
            r=16,                         # rank
            lora_alpha=32,
            target_modules=["q", "v"],    # VERY IMPORTANT for T5
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)

        # Optional: print trainable params
        self.model.print_trainable_parameters()

        self.rouge = evaluate.load("rouge")

    def split_dataset(self):
        # Slice the first 50,000 rows before splitting
        # This reduces training time
        self.dataset["train"] = self.dataset["train"].select(range(50000))
        
        split = self.dataset["train"].train_test_split(test_size=0.05)
        self.train_dataset = split["train"]
        self.eval_dataset = split["test"]

    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        self.split_dataset()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=int(self.config.num_train_epochs),
            warmup_steps=int(self.config.warmup_steps),

            # You can increase now (LoRA advantage)
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,

            weight_decay=float(self.config.weight_decay),
            logging_steps=int(self.config.logging_steps),

            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,

            gradient_accumulation_steps=2,

            fp16=torch.cuda.is_available(),

            # Fast generation settings
            generation_max_length=128,
            generation_num_beams=1,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=None,   # keep OFF for speed
        )

        trainer.train()

        # Optional evaluation after training
        trainer.predict(self.eval_dataset, max_new_tokens=128)

        # Save LoRA adapters ONLY (important)
        self.model.save_pretrained(self.config.root_dir)

        self.tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )

        logger.info("LoRA Training complete!")