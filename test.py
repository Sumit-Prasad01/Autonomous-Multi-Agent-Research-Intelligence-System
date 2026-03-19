import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def compare_summaries(abstract_text):
    model_id = "google/flan-t5-small"
    
    # 1. Load Original Model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Load Your Fine-tuned Model (LoRA)
    # OUTPUT_DIR is where you saved your model earlier
    fine_tuned_model = PeftModel.from_pretrained(base_model, "/content/drive/MyDrive/Finetune_model/")
    
    inputs = tokenizer("summarize: " + abstract_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    # Generate from Base
    with torch.no_grad():
        base_output = base_model.generate(**inputs, max_new_tokens=150)
        ft_output = fine_tuned_model.generate(**inputs, max_new_tokens=150)
        
    print(f"\n--- ORIGINAL ABSTRACT ---\n{abstract_text[:300]}...")
    print(f"\n--- BASE MODEL SUMMARY ---\n{tokenizer.decode(base_output[0], skip_special_tokens=True)}")
    print(f"\n--- YOUR FINE-TUNED SUMMARY ---\n{tokenizer.decode(ft_output[0], skip_special_tokens=True)}")

# Test it!
sample_abstract = "Insert a real abstract from your dataset here"
compare_summaries(sample_abstract)