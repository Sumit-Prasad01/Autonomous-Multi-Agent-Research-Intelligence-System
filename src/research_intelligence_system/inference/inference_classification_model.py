# =========================
# Inference Pipeline
# flan-t5-small + LoRA
# Research Abstract Summarization
# =========================

# !pip install -q transformers peft torch

# =========================
# 1. Imports
# =========================
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from typing import Union

# =========================
# 2. Config
# =========================
BASE_MODEL     = "google/flan-t5-small"
LORA_CKPT      = "/content/drive/MyDrive/finetuned_model"
TOKENIZER_PATH = os.path.join(LORA_CKPT, "tokenizer")

MAX_INPUT_LEN  = 256
MAX_NEW_TOKENS = 128
NUM_BEAMS      = 2

# =========================
# 3. Summarizer Class
# =========================
class AbstractSummarizer:
    def __init__(
        self,
        base_model: str = BASE_MODEL,
        lora_ckpt: str  = LORA_CKPT,
        tokenizer_path: str = TOKENIZER_PATH,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading on: {self.device}")

        print("📦 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        print("🤖 Loading model...")
        base = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model = PeftModel.from_pretrained(base, lora_ckpt)
        self.model.eval()
        print("✅ Model ready\n")

    # --------------------------------------------------
    # Core summarize method
    # --------------------------------------------------
    def summarize(
        self,
        text: Union[str, list],
        max_new_tokens: int = MAX_NEW_TOKENS,
        num_beams: int      = NUM_BEAMS,
        max_input_len: int  = MAX_INPUT_LEN,
    ) -> Union[str, list]:
        """
        Summarize a single abstract string or a list of strings.
        Returns a string if input is a string, list if input is a list.
        """
        single = isinstance(text, str)
        texts  = [text] if single else text

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_input_len,
            truncation=True,
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,     # 1.0 = neutral, >1 = longer, <1 = shorter
            )

        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return summaries[0] if single else summaries

    # --------------------------------------------------
    # Summarize from a .txt file (one abstract per file)
    # --------------------------------------------------
    def summarize_file(self, filepath: str, **kwargs) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return self.summarize(text, **kwargs)

    # --------------------------------------------------
    # Batch summarize from a list of files
    # --------------------------------------------------
    def summarize_files(self, filepaths: list, **kwargs) -> list:
        texts = []
        for fp in filepaths:
            with open(fp, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())
        return self.summarize(texts, **kwargs)

    # --------------------------------------------------
    # Summarize a CSV column of abstracts
    # --------------------------------------------------
    def summarize_csv(
        self,
        csv_path: str,
        text_col: str   = "abstract",
        output_col: str = "summary",
        save_path: str  = None,
        batch_size: int = 8,
    ):
        import pandas as pd

        print(f"📄 Reading {csv_path}...")
        df = pd.read_csv(csv_path)
        assert text_col in df.columns, f"Column '{text_col}' not found. Available: {list(df.columns)}"

        texts = df[text_col].fillna("").tolist()
        summaries = []

        print(f"⚙️  Summarizing {len(texts)} rows in batches of {batch_size}...")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            summaries.extend(self.summarize(batch))
            print(f"   {min(i+batch_size, len(texts))}/{len(texts)} done")

        df[output_col] = summaries

        out = save_path or csv_path.replace(".csv", "_summarized.csv")
        df.to_csv(out, index=False)
        print(f"✅ Saved to {out}")
        return df

    # --------------------------------------------------
    # Interactive mode — type abstracts in the terminal
    # --------------------------------------------------
    def interactive(self):
        print("="*55)
        print("  Abstract Summarizer — interactive mode")
        print("  Type or paste an abstract. Enter 'quit' to exit.")
        print("="*55)
        while True:
            print("\nAbstract (paste, then press Enter twice):")
            lines = []
            while True:
                line = input()
                if line.lower() == "quit":
                    print("Goodbye!")
                    return
                if line == "" and lines:
                    break
                lines.append(line)

            text = " ".join(lines).strip()
            if not text:
                continue

            print("\n⚙️  Summarizing...")
            summary = self.summarize(text)
            print(f"\n📝 Summary:\n  {summary}\n")
            print("-"*55)


# =========================
# 4. Usage Examples
# =========================
if __name__ == "__main__":

    # --- Load once, reuse many times ---
    summarizer = AbstractSummarizer()

    # -----------------------------------------
    # Example 1: single abstract string
    # -----------------------------------------
    abstract = """
    Large language models (LLMs) have demonstrated remarkable capabilities across
    a wide range of natural language processing tasks. However, their deployment
    in resource-constrained environments remains challenging due to high memory
    and compute requirements. In this paper, we propose a novel parameter-efficient
    fine-tuning approach that combines low-rank adaptation with quantization-aware
    training, achieving competitive performance on summarization benchmarks while
    reducing memory footprint by 60% compared to full fine-tuning baselines.
    Experiments on PubMed and arXiv datasets confirm the effectiveness of our method.
    """

    summary = summarizer.summarize(abstract)
    print("Single abstract:")
    print(f"  Input : {abstract[:120].strip()}...")
    print(f"  Output: {summary}\n")

    # -----------------------------------------
    # Example 2: batch of abstracts
    # -----------------------------------------
    abstracts = [
        "We present a new transformer architecture optimized for long document understanding...",
        "This study investigates the effect of learning rate scheduling on fine-tuning stability...",
        "We propose a contrastive learning framework for scientific text representation...",
    ]

    summaries = summarizer.summarize(abstracts)
    print("Batch abstracts:")
    for i, s in enumerate(summaries):
        print(f"  [{i+1}] {s}")

    # -----------------------------------------
    # Example 3: summarize a CSV file
    # -----------------------------------------
    # summarizer.summarize_csv(
    #     csv_path="papers.csv",
    #     text_col="abstract",     # column name containing abstracts
    #     output_col="summary",    # new column to write summaries into
    #     batch_size=8,
    # )

    # -----------------------------------------
    # Example 4: summarize a .txt file
    # -----------------------------------------
    # summary = summarizer.summarize_file("abstract.txt")
    # print(summary)

    # -----------------------------------------
    # Example 5: interactive terminal mode
    # -----------------------------------------
    # summarizer.interactive()
