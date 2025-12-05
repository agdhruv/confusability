"""
GSM-8k Noise Experiment

Fine-tune models on GSM-8k with varying levels of label noise.
Hypothesis: do bigger models lose capabilities more rapidly with noisy data?
"""
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lm_eval
import torch
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments

from src.evals import InmemoryPeftLM
from src.noise import number_perturbation
from src.utils import load_model, load_tokenizer, to_lora

# ============== CONFIG ==============
MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
]
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_EPOCHS = 3
OUTPUT_DIR = Path("./results/gsm8k_noise")
MAX_SEQ_LEN = 512
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EVAL_BATCH_SIZE = 64
EVAL_LIMIT = None  # e.g. 100 for debugging
_gsm8k_path = Path(lm_eval.__file__).parent / "tasks" / "gsm8k"
TASK_MANAGER = TaskManager(include_defaults=False, include_path=str(_gsm8k_path))
# ====================================


@dataclass
class ExperimentResult:
    model_id: str
    noise_level: float
    epoch: int
    post_train_accuracy: float
    train_loss: float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_label": self.model_id.split("/")[-1],
            "noise_level": self.noise_level,
            "epoch": self.epoch,
            "post_train_accuracy": self.post_train_accuracy,
            "train_loss": self.train_loss,
            "timestamp": self.timestamp,
        }


def evaluate_gsm8k(model, tokenizer, model_id: str) -> float:
    """Evaluate on GSM-8k test set."""
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    lm = InmemoryPeftLM(model, tokenizer, model_id, device=str(model.device), batch_size=EVAL_BATCH_SIZE)
    with torch.inference_mode():
        results = simple_evaluate( # type: ignore
            model=lm,
            task_manager=TASK_MANAGER,
            tasks=["gsm8k"],
            num_fewshot=0,
            batch_size=EVAL_BATCH_SIZE,
            verbosity="ERROR",
            limit=EVAL_LIMIT,
            cache_requests=True,
            gen_kwargs={"max_new_tokens": None},
        )
    gsm8k = results["results"]["gsm8k"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return gsm8k.get("exact_match,strict-match", gsm8k.get("acc,none", 0))


def prepare_noised_dataset(tokenizer, noise_level: float, seed: int = 42):
    """Load GSM-8k and apply label noise."""
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    
    rng = random.Random(seed)
    n_noised = int(len(train_ds) * noise_level)
    noise_indices = set(rng.sample(range(len(train_ds)), n_noised)) if n_noised else set()

    def tokenize(examples, indices):
        texts = []
        for idx, q, a in zip(indices, examples["question"], examples["answer"]):
            if idx in noise_indices:
                a = number_perturbation(a, rng, scale=0.5)
            texts.append(f"Question: {q}\nAnswer: {a}")
        out = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    return train_ds.map(tokenize, batched=True, with_indices=True, remove_columns=train_ds.column_names)


def load_lora_model(model_id: str):
    """Load model with LoRA adapter."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = load_model(model_id, dtype)
    return to_lora(model, checkpointing=True)


class EpochEvalCallback(TrainerCallback):
    """Evaluate on GSM8k at the end of each epoch."""
    
    def __init__(self, model, tokenizer, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.epoch_results: list[tuple[int, float, float]] = []  # (epoch, accuracy, loss)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        # Get loss from most recent log entry
        loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), 0.0)
        
        print(f"\n--- End of Epoch {epoch}/{NUM_EPOCHS} - Evaluating... ---")
        acc = evaluate_gsm8k(self.model, self.tokenizer, self.model_id)
        print(f"Epoch {epoch} accuracy: {acc:.2%}")
        
        self.epoch_results.append((epoch, acc, loss))


def run_experiment(model_id: str, noise_level: float) -> list[ExperimentResult]:
    """Single experiment: train on noised data, evaluate after each epoch."""
    print(f"\n{'='*60}")
    print(f"{model_id.split('/')[-1]} | noise={noise_level:.0%}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(model_id)
    model = load_lora_model(model_id)
    train_dataset = prepare_noised_dataset(tokenizer, noise_level)
    
    eval_callback = EpochEvalCallback(model, tokenizer, model_id)
    
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"{model_id.split('/')[-1]}_noise{noise_level:.2f}"),
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
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[eval_callback],
    )
    trainer.train()
    
    # Convert callback results to ExperimentResults
    results = [
        ExperimentResult(
            model_id=model_id,
            noise_level=noise_level,
            epoch=epoch,
            post_train_accuracy=acc,
            train_loss=loss,
            timestamp=datetime.now().isoformat(),
        )
        for epoch, acc, loss in eval_callback.epoch_results
    ]

    del model
    torch.cuda.empty_cache()
    return results


def load_checkpoint() -> tuple[list[dict], set[tuple[str, float]]]:
    """Load existing results, return (results, completed_set). Only complete if all epochs done."""
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        return [], set()
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Count epochs per (model, noise) pair - only complete if all NUM_EPOCHS done
    from collections import Counter
    epoch_counts = Counter((r["model_id"], r["noise_level"]) for r in results)
    completed = {k for k, v in epoch_counts.items() if v >= NUM_EPOCHS}
    
    # Remove partial results (incomplete epoch runs)
    results = [r for r in results if (r["model_id"], r["noise_level"]) in completed]
    
    print(f"Resuming: {len(completed)} full runs complete ({len(results)} epoch-results)")
    return results, completed


def save_checkpoint(results: list[dict]):
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    torch.manual_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results, completed = load_checkpoint()

    for model_id in MODELS:
        for noise_level in NOISE_LEVELS:
            if (model_id, noise_level) in completed:
                print(f"Skipping {model_id} @ {noise_level:.0%} (done)")
                continue
            
            epoch_results = run_experiment(model_id, noise_level)
            for r in epoch_results:
                results.append(r.to_dict())
            completed.add((model_id, noise_level))
            save_checkpoint(results)

    # Final summary
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'Noise':<8} {'Epoch':<6} {'Accuracy':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['model_id'].split('/')[-1]:<20} {r['noise_level']:<8.0%} {r['epoch']:<6} {r['post_train_accuracy']:<10.2%}")


if __name__ == "__main__":
    main()
