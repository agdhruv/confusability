"""
GSM-8k Noise Experiment

Fine-tune models on GSM-8k with varying levels of label noise.
Hypothesis: do bigger models lose capabilities more rapidly with noisy data?
"""
import os
import json
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from lm_eval import simple_evaluate

from src.utils import load_tokenizer, load_model, to_lora
from src.evals import InmemoryPeftLM
from src.noise import number_perturbation

torch.random.manual_seed(42)

# ============== CONFIG ==============
MODELS = [
    "Qwen/Qwen2.5-7B",
]
NOISE_LEVELS = [0.1, 0.2]
OUTPUT_DIR = Path("./results/gsm8k_noise")
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EVAL_BATCH_SIZE = 8
EVAL_LIMIT = None  # Set to e.g. 100 for debugging
# ====================================


def load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main")
    return ds["train"], ds["test"]


def prepare_dataset(train_ds, tokenizer, noise_level: float, seed: int = 42):
    """Apply noise and tokenize."""
    rng = random.Random(seed)
    n_samples = len(train_ds)
    n_noised = int(n_samples * noise_level)
    noise_indices = set(rng.sample(range(n_samples), n_noised)) if n_noised > 0 else set()
    
    def process_and_tokenize(examples, indices):
        breakpoint()
        texts = []
        for idx, q, a in zip(indices, examples["question"], examples["answer"]):
            if idx in noise_indices:
                a = number_perturbation(a, rng)
            texts.append(f"Question: {q}\nAnswer: {a}")
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    return train_ds.map(
        process_and_tokenize,
        batched=True,
        with_indices=True,
        remove_columns=train_ds.column_names,
    ), len(noise_indices)


def run_single_experiment(model_id: str, noise_level: float) -> dict:
    """Fine-tune on noised GSM-8k, evaluate on clean test."""
    model_label = model_id.split('/')[-1]
    run_name = f"{model_label}_noise{noise_level:.2f}"
    
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"{'='*60}\n")
    
    tokenizer = load_tokenizer(model_id)
    train_ds, _ = load_gsm8k()
    
    train_tokenized, n_noised = prepare_dataset(train_ds, tokenizer, noise_level)
    breakpoint()
    print(f"Training samples: {len(train_ds)}, Noised: {n_noised}")
    
    # Load model with LoRA
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = load_model(model_id, torch_dtype)
    model = to_lora(model, checkpointing=True)
    model.print_trainable_parameters()
    
    grad_accum = BATCH_SIZE // MICRO_BATCH_SIZE
    steps_per_epoch = len(train_tokenized) // BATCH_SIZE
    
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / run_name),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        max_steps=steps_per_epoch,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        save_strategy="no",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    train_result = trainer.train()
    train_loss = train_result.training_loss
    print(f"Training loss: {train_loss:.4f}")
    
    # Evaluate
    print("Evaluating on GSM-8k test set...")
    lm_object = InmemoryPeftLM(model, tokenizer, model_id, device=str(model.device), batch_size=EVAL_BATCH_SIZE)
    
    eval_results = simple_evaluate(
        model=lm_object,
        tasks=["gsm8k"],
        num_fewshot=5,
        batch_size=EVAL_BATCH_SIZE,
        verbosity="ERROR",
        limit=EVAL_LIMIT,
    )
    
    gsm8k_results = eval_results["results"]["gsm8k"]
    accuracy = gsm8k_results.get("exact_match,strict-match", gsm8k_results.get("acc,none", 0))
    print(f"GSM-8k Accuracy: {accuracy:.2%}")
    
    del model, trainer
    torch.cuda.empty_cache()
    
    return {
        "model_id": model_id,
        "model_label": model_label,
        "noise_level": noise_level,
        "train_loss": train_loss,
        "gsm8k_accuracy": float(accuracy),
        "gsm8k_results": gsm8k_results,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    for model_id in MODELS:
        for noise_level in NOISE_LEVELS:
            try:
                result = run_single_experiment(model_id, noise_level)
                all_results.append(result)
                
                # Save after each run
                with open(OUTPUT_DIR / "results.json", "w") as f:
                    json.dump(all_results, f, indent=2)
            except Exception as e:
                print(f"Error: {model_id} @ noise={noise_level}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Model':<20} {'Noise':<10} {'Accuracy':<10}")
    print("-"*40)
    for r in all_results:
        print(f"{r['model_label']:<20} {r['noise_level']:<10.0%} {r['gsm8k_accuracy']:<10.2%}")


if __name__ == "__main__":
    main()
