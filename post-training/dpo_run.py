"""
Train a DPO model starting from SFT checkpoint.
"""
import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ============== CONFIG ==============
# Start from your robust SFT ingredient
MODEL_ID = "./results/alpaca_soup/Qwen2.5-0.5B/ingredient_0" 
OUTPUT_DIR = Path("./results/dpo_experiment")
NUM_EPOCHS = 1 
LEARNING_RATE = 5e-7            # DPO usually needs very low LR
BETA = 0.1                      # The KL penalty strength (standard is 0.1)
BATCH_SIZE = 16
GRAD_ACCUM = 2
# ====================================

def main():
    print(f"Starting DPO training from: {MODEL_ID}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Preference Data
    # We use a subset of UltraFeedback, a standard DPO dataset
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:50000]") # Subset for speed
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["chosen", "rejected"]])
    
    # 3. Load Model (Policy)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # 4. Configure DPO
    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        beta=BETA,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to="none",
        max_length=1024,
        max_prompt_length=512,
    )

    # 5. Train
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # None = load a copy of 'model' as reference
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Training...")
    trainer.train()
    
    # 6. Save
    final_path = OUTPUT_DIR / "dpo_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"DPO Model saved to {final_path}")

if __name__ == "__main__":
    main()