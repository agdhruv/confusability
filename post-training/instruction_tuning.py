"""
Instruction Tuning with Ultrachat

Fine-tune base models on the Ultrachat dataset using TRL's SFTTrainer.
The resulting models can then be used with gsm8k_noise_run.py or arc_noise_run.py
for noise experiments.

Usage:
    python instruction_tuning.py                    # Train all models
    python instruction_tuning.py --model Qwen/Qwen2.5-0.5B  # Train specific model
"""
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from src.utils import load_model, load_tokenizer

# ============== CONFIG ==============
BASE_MODELS = [
    "Qwen/Qwen2.5-0.5B",
]
OUTPUT_DIR = Path("./results/instruction_tuning")
NUM_EPOCHS = 1
MAX_SEQ_LEN = 2048
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-5
DATASET_SIZE = 50_000  # Subset of Ultrachat to use (None for full dataset)
# ====================================


def format_ultrachat(example):
    """Convert messages to prompt-completion format for completion_only_loss."""
    messages = example["messages"]
    # All messages except last are prompt, last assistant message is completion
    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
    }


def load_ultrachat(max_samples: int | None = None):
    """Load and prepare Ultrachat dataset."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    
    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    
    dataset = dataset.map(format_ultrachat, remove_columns=dataset.column_names)
    return dataset


def train_model(model_id: str):
    """Fine-tune a single model on Ultrachat."""
    out_dir = OUTPUT_DIR / model_id.split("/")[-1]
    
    if out_dir.exists():
        print(f"Model already exists at {out_dir}, skipping.")
        return
    
    print(f"\n{'='*60}")
    print(f"Instruction tuning: {model_id}")
    print(f"{'='*60}")
    
    tokenizer = load_tokenizer(model_id)
    
    # Load dataset
    dataset = load_ultrachat(max_samples=DATASET_SIZE)
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = load_model(model_id, dtype)
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LEN,
        completion_only_loss=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    trainer.train()
    
    # Merge LoRA weights and save
    print(f"Merging and saving model to {out_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(out_dir)
    print(f"Saved instruction-tuned model to {out_dir}")
    
    del model, merged_model, trainer
    torch.cuda.empty_cache()


def main():    
    torch.manual_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_to_train = BASE_MODELS
    
    for model_id in models_to_train:
        train_model(model_id)
    
    print("\n" + "="*60)
    print("Instruction tuning complete!")
    print("Models saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()

