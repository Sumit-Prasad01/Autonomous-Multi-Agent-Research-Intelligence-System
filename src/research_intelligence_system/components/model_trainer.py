from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
import torch
from datasets import load_from_disk
import os
from src.research_intelligence_system.entity import ModelTrainerConfig
from src.research_intelligence_system.utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _get_device_and_dtype(self):
        """Optimized device setup for RTX 3050 4GB VRAM."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} | VRAM: {vram_gb:.1f} GB")
            torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on Ampere
            torch.backends.cudnn.allow_tf32 = True

            # bf16 is stable on Ampere (RTX 3050); fp16 only as fallback
            use_bf16 = torch.cuda.is_bf16_supported()
            use_fp16 = not use_bf16
            logger.info(f"Precision: {'bf16' if use_bf16 else 'fp16'}")
            return "cuda", use_fp16, use_bf16
        else:
            logger.info("No GPU found, using CPU.")
            return "cpu", False, False

    def train(self):
        device, use_fp16, use_bf16 = self._get_device_and_dtype()

        # ── Hardcoded values (not present in ModelTrainerConfig) ──────────────
        EARLY_STOP_PATIENCE = 3         # stop if eval_loss doesn't improve for 3 evals
        VAL_SPLIT_SIZE      = 0.1       # 10% of train data used as validation
        VAL_SPLIT_SEED      = 42        # reproducible split
        # ─────────────────────────────────────────────────────────────────────

        # Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None  # tensor core alignment
        )

        # ── Load dataset & split train → train/val ────────────────────────────
        dataset = load_from_disk(self.config.data_path)
        logger.info("Dataset columns    :", dataset["train"].column_names)
        logger.info(f"Original train size: {len(dataset['train'])}")

        split = dataset["train"].train_test_split(
            test_size=VAL_SPLIT_SIZE,
            seed=VAL_SPLIT_SEED
        )
        train_dataset = split["train"]
        val_dataset   = split["test"]  
        logger.info(f"After split → Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        # ─────────────────────────────────────────────────────────────────────

        trainer_args = TrainingArguments(
            # ── From ModelTrainerConfig ───────────────────────────────────────
            output_dir=self.config.root_dir,
            num_train_epochs=int(self.config.num_train_epochs),
            warmup_steps=int(self.config.warmup_steps),
            per_device_train_batch_size=int(self.config.per_device_train_batch_size),   
            per_device_eval_batch_size=int(self.config.per_device_train_batch_size),
            weight_decay=float(self.config.weight_decay),
            logging_steps=int(self.config.logging_steps),
            gradient_accumulation_steps=int(self.config.gradient_accumulation_steps),  

            # ── Strategy (must match for load_best_model_at_end) ──────────────
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # ── Optimizer & LR ────────────────────────────────────────────────
            optim="adamw_torch",        # respects warmup & scheduler unlike adafactor
            learning_rate=3e-4,
            lr_scheduler_type="linear", # warmup → linear decay
            max_grad_norm=1.0,          # clips exploding gradients → prevents NaN

            # ── Precision (bf16 > fp16 on Ampere — no overflow issues) ────────
            bf16=use_bf16,              # primary: stable, same range as float32
            fp16=use_fp16,              # fallback only if bf16 not supported

            # ── Memory optimizations for 4GB VRAM ─────────────────────────────
            gradient_checkpointing=True,        # recompute activations to save VRAM
            dataloader_pin_memory=False,        # reduces CPU→GPU memory overhead
            report_to="none",                   # disables wandb
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            processing_class=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=EARLY_STOP_PATIENCE
                )
            ]
        )

        trainer.train()

        # Save final model & tokenizer
        model.save_pretrained(os.path.join(self.config.root_dir, "flan-t5-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
        logger.info("Training complete. Model saved.")