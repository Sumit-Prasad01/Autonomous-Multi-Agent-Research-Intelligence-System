import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def compare_summaries(abstract_text):
    model_id = "google/flan-t5-small"
    lora_path = "artifacts/model_trainer"   # update if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # 1. Load tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # =========================
    # 2. Load BASE model (clean)
    # =========================
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    base_model.eval()

    # =========================
    # 3. Load Fine-tuned model (separate base)
    # =========================
    ft_base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    fine_tuned_model = PeftModel.from_pretrained(ft_base_model, lora_path)
    fine_tuned_model.eval()

    # =========================
    # 4. Prepare input
    # =========================
    prompt = "summarize: " + abstract_text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # =========================
    # 5. Generate (IMPROVED)
    # =========================
    gen_kwargs = dict(
        max_new_tokens=120,
        num_beams=4,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    with torch.no_grad():
        base_output = base_model.generate(**inputs, **gen_kwargs)
        ft_output = fine_tuned_model.generate(**inputs, **gen_kwargs)

    base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
    ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)

    # =========================
    # 6. Print Results
    # =========================
    print("\n" + "="*60)
    print("🔍 COMPARISON")
    print("="*60)

    print(f"\n📄 INPUT (truncated):\n{abstract_text[:300]}...\n")

    print("🟡 BASE MODEL OUTPUT:")
    print(base_text if base_text else "❌ EMPTY OUTPUT")

    print("\n🟢 FINE-TUNED MODEL OUTPUT:")
    print(ft_text if ft_text else "❌ EMPTY OUTPUT")

    print("\n" + "="*60)


# =========================
# 🔥 TEST
# =========================
sample_abstract = """
Deep learning models have achieved state-of-the-art performance in various NLP tasks.
However, they require large amounts of labeled data and computational resources.
This paper explores efficient fine-tuning techniques using low-rank adaptation (LoRA)
to reduce memory and computational overhead while maintaining performance.
"""

compare_summaries(sample_abstract)