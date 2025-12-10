"""
Transfer Learning Noise Experiment

Fine-tune base models on one dataset (GSM8k or ARC), then run noise experiments
on the other dataset to study transfer learning under label noise.

Flow:
1. Fine-tune base model on source dataset (e.g., GSM8k)
2. Merge LoRA weights and save the fine-tuned model
3. Load fine-tuned model, apply LoRA, and train on target dataset with noise
"""
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lm_eval
import torch
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.evals import InmemoryPeftLM
from src.noise import number_perturbation
from src.utils import load_model, load_tokenizer, to_lora

# ============== CONFIG ==============
BASE_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3",
]
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_PRETRAIN_EPOCHS = 3
NUM_NOISE_EPOCHS = 3
OUTPUT_DIR = Path("./results/transfer_noise")
FINETUNED_MODELS_DIR = Path("./results/transfer_noise/models/finetuned")
MAX_SEQ_LEN = 512
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EVAL_BATCH_SIZE = 128
EVAL_LIMIT = None  # e.g. 100 for debugging

# Task managers for evaluation
_arc_path = Path(lm_eval.__file__).parent / "tasks" / "arc"
_gsm8k_path = Path(lm_eval.__file__).parent / "tasks" / "gsm8k"
ARC_TASK_MANAGER = TaskManager(include_defaults=False, include_path=str(_arc_path))
GSM8K_TASK_MANAGER = TaskManager(include_defaults=False, include_path=str(_gsm8k_path))
# ====================================


class Dataset(Enum):
    GSM8K = "gsm8k"
    ARC = "arc"


@dataclass
class ExperimentResult:
    model_id: str
    source_dataset: str
    target_dataset: str
    noise_level: float
    epoch: int
    accuracy: float
    train_loss: float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_label": self.model_id.split("/")[-1],
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "noise_level": self.noise_level,
            "epoch": self.epoch,
            "accuracy": self.accuracy,
            "train_loss": self.train_loss,
            "timestamp": self.timestamp,
        }


# ============== EVALUATION ==============


def evaluate_arc(model, tokenizer, model_id: str) -> float:
    """Evaluate on ARC-Challenge (base model format)."""
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    lm = InmemoryPeftLM(model, tokenizer, model_id, device=str(model.device), batch_size=EVAL_BATCH_SIZE)
    with torch.inference_mode():
        results = simple_evaluate( # type: ignore
            model=lm,
            task_manager=ARC_TASK_MANAGER,
            tasks=["arc_challenge"],
            num_fewshot=0,
            batch_size=EVAL_BATCH_SIZE,
            verbosity="ERROR",
            limit=EVAL_LIMIT,
            cache_requests=True,
            gen_kwargs={"max_new_tokens": None},
            apply_chat_template=False,
        )
    arc = results["results"]["arc_challenge"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return arc.get("acc_norm,none", arc.get("acc,none", 0))


def evaluate_gsm8k(model, tokenizer, model_id: str) -> float:
    """Evaluate on GSM-8k test set (base model format)."""
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    lm = InmemoryPeftLM(model, tokenizer, model_id, device=str(model.device), batch_size=EVAL_BATCH_SIZE)
    with torch.inference_mode():
        results = simple_evaluate( # type: ignore
            model=lm,
            task_manager=GSM8K_TASK_MANAGER,
            tasks=["gsm8k"],
            num_fewshot=0,
            batch_size=EVAL_BATCH_SIZE,
            verbosity="ERROR",
            limit=EVAL_LIMIT,
            cache_requests=True,
            gen_kwargs={"max_new_tokens": None},
            apply_chat_template=False,
        )
    gsm8k = results["results"]["gsm8k"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return gsm8k.get("exact_match,strict-match", gsm8k.get("acc,none", 0))


def evaluate(model, tokenizer, model_id: str, dataset: Dataset) -> float:
    """Evaluate on specified dataset."""
    if dataset == Dataset.ARC:
        return evaluate_arc(model, tokenizer, model_id)
    else:
        return evaluate_gsm8k(model, tokenizer, model_id)


# ============== DATA PREPARATION ==============


def format_gsm8k_example(question: str, answer: str) -> str:
    """Format GSM8k example for base model (no chat template)."""
    return f"Question: {question}\nAnswer: {answer}"


def format_arc_example(question: str, answer_text: str) -> str:
    """Format ARC example for base model (no chat template)."""
    return f"Question: {question}\nAnswer: {answer_text}"


def prepare_gsm8k_dataset(tokenizer, noise_level: float = 0.0, seed: int = 42):
    """Load GSM-8k and optionally apply label noise."""
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    
    rng = random.Random(seed)
    n_noised = int(len(train_ds) * noise_level)
    noise_indices = set(rng.sample(range(len(train_ds)), n_noised)) if n_noised else set()

    def tokenize(examples, indices):
        texts = []
        for idx, q, a in zip(indices, examples["question"], examples["answer"]):
            if idx in noise_indices:
                a = number_perturbation(a, rng, scale=0.5)
            texts.append(format_gsm8k_example(q, a))
        out = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    return train_ds.map(tokenize, batched=True, with_indices=True, remove_columns=train_ds.column_names)


def prepare_arc_dataset(tokenizer, noise_level: float = 0.0, seed: int = 42):
    """Load ARC-Challenge and optionally apply label noise."""
    train_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    
    rng = random.Random(seed)
    n_noised = int(len(train_ds) * noise_level)
    noise_indices = set(rng.sample(range(len(train_ds)), n_noised)) if n_noised else set()

    def tokenize(examples, indices):
        texts = []
        for idx, q, choices, answer_key in zip(indices, examples["question"], examples["choices"], examples["answerKey"]):
            choice_texts = choices["text"]
            choice_labels = choices["label"]
            correct_idx = choice_labels.index(answer_key)

            noisy_idx = correct_idx
            if idx in noise_indices:
                candidates = [i for i in range(len(choice_texts)) if i != correct_idx]
                noisy_idx = rng.choice(candidates)

            noisy_letter = choice_labels[noisy_idx]
            noisy_text = choice_texts[noisy_idx]
            texts.append(format_arc_example(q, noisy_text))

        out = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    return train_ds.map(tokenize, batched=True, with_indices=True, remove_columns=train_ds.column_names)


def prepare_dataset(tokenizer, dataset: Dataset, noise_level: float = 0.0, seed: int = 42):
    """Prepare specified dataset with optional noise."""
    if dataset == Dataset.GSM8K:
        return prepare_gsm8k_dataset(tokenizer, noise_level, seed)
    else:
        return prepare_arc_dataset(tokenizer, noise_level, seed)


# ============== MODEL UTILITIES ==============


def load_lora_model(model_id: str):
    """Load model with LoRA adapter."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = load_model(model_id, dtype)
    return to_lora(model, checkpointing=True)


def get_finetuned_model_path(model_id: str, dataset: Dataset) -> Path:
    """Get path to save/load fine-tuned model."""
    model_name = model_id.split("/")[-1]
    return FINETUNED_MODELS_DIR / f"{model_name}_{dataset.value}"


def save_merged_model(model, save_path: Path):
    """Merge LoRA weights and save the full model."""
    print(f"Merging and saving model to {save_path}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    print(f"Saved merged model to {save_path}")


def load_finetuned_model(model_path: Path):
    """Load a previously fine-tuned model and apply LoRA for further training."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = load_model(str(model_path), dtype)
    return to_lora(model, checkpointing=True)


# ============== TRAINING CALLBACKS ==============


class EpochEvalCallback(TrainerCallback):
    """Evaluate at the end of each epoch."""
    
    def __init__(self, model, tokenizer, model_id: str, dataset: Dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.dataset = dataset
        self.epoch_results: list[tuple[int, float, float]] = []  # (epoch, accuracy, loss)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), 0.0)
        
        print(f"\n--- End of Epoch {epoch} - Evaluating on {self.dataset.value}... ---")
        acc = evaluate(self.model, self.tokenizer, self.model_id, self.dataset)
        print(f"Epoch {epoch} accuracy: {acc:.2%}")
        
        self.epoch_results.append((epoch, acc, loss))


# ============== TRAINING FUNCTIONS ==============


def pretrain_on_source(model_id: str, source_dataset: Dataset) -> Path:
    """
    Fine-tune base model on source dataset.
    Returns path to saved merged model.
    """
    save_path = get_finetuned_model_path(model_id, source_dataset)
    
    if save_path.exists():
        print(f"Found existing fine-tuned model at {save_path}, skipping pre-training.")
        return save_path
    
    print(f"\n{'='*60}")
    print(f"Pre-training {model_id.split('/')[-1]} on {source_dataset.value}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(model_id)
    model = load_lora_model(model_id)
    
    train_dataset = prepare_dataset(tokenizer, source_dataset, noise_level=0.0)
    
    output_dir = OUTPUT_DIR / f"pretrain_{model_id.split('/')[-1]}_{source_dataset.value}"
    
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        num_train_epochs=NUM_PRETRAIN_EPOCHS,
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
    )
    
    trainer.train()
    
    # Save merged model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_merged_model(model, save_path)
    
    del model
    torch.cuda.empty_cache()
    
    return save_path


def run_noise_experiment(
    finetuned_model_path: Path,
    original_model_id: str,
    source_dataset: Dataset,
    target_dataset: Dataset,
    noise_level: float,
) -> list[ExperimentResult]:
    """
    Run noise experiment: train fine-tuned model on target dataset with noise.
    """
    print(f"\n{'='*60}")
    print(f"{original_model_id.split('/')[-1]} | {source_dataset.value} -> {target_dataset.value} | noise={noise_level:.0%}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(original_model_id)
    model = load_finetuned_model(finetuned_model_path)
    
    train_dataset = prepare_dataset(tokenizer, target_dataset, noise_level)
    eval_callback = EpochEvalCallback(model, tokenizer, original_model_id, target_dataset)
    
    output_dir = OUTPUT_DIR / f"{original_model_id.split('/')[-1]}_{source_dataset.value}_to_{target_dataset.value}_noise{noise_level:.2f}"
    
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        num_train_epochs=NUM_NOISE_EPOCHS,
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
    
    results = [
        ExperimentResult(
            model_id=original_model_id,
            source_dataset=source_dataset.value,
            target_dataset=target_dataset.value,
            noise_level=noise_level,
            epoch=epoch,
            accuracy=acc,
            train_loss=loss,
            timestamp=datetime.now().isoformat(),
        )
        for epoch, acc, loss in eval_callback.epoch_results
    ]

    del model
    torch.cuda.empty_cache()
    return results


# ============== CHECKPOINTING ==============


def load_checkpoint(direction: str) -> tuple[list[dict], set[tuple[str, float]]]:
    """Load existing results for a given direction (gsm8k_to_arc or arc_to_gsm8k)."""
    results_path = OUTPUT_DIR / f"results_{direction}.json"
    if not results_path.exists():
        return [], set()
    
    with open(results_path) as f:
        results = json.load(f)
    
    from collections import Counter
    epoch_counts = Counter((r["model_id"], r["noise_level"]) for r in results)
    completed = {k for k, v in epoch_counts.items() if v >= NUM_NOISE_EPOCHS}
    
    results = [r for r in results if (r["model_id"], r["noise_level"]) in completed]
    
    print(f"Resuming {direction}: {len(completed)} full runs complete ({len(results)} epoch-results)")
    return results, completed


def save_checkpoint(results: list[dict], direction: str):
    """Save results for a given direction."""
    with open(OUTPUT_DIR / f"results_{direction}.json", "w") as f:
        json.dump(results, f, indent=2)


# ============== MAIN EXPERIMENT RUNNERS ==============


def run_gsm8k_to_arc():
    """Fine-tune on GSM8k, then run noise experiments on ARC."""
    direction = "gsm8k_to_arc"
    results, completed = load_checkpoint(direction)
    
    for model_id in BASE_MODELS:
        # Pre-train on GSM8k
        finetuned_path = pretrain_on_source(model_id, Dataset.GSM8K)
        
        # Run noise experiments on ARC
        for noise_level in NOISE_LEVELS:
            if (model_id, noise_level) in completed:
                print(f"Skipping {model_id} @ {noise_level:.0%} (done)")
                continue
            
            epoch_results = run_noise_experiment(
                finetuned_path, model_id, Dataset.GSM8K, Dataset.ARC, noise_level
            )
            for r in epoch_results:
                results.append(r.to_dict())
            completed.add((model_id, noise_level))
            save_checkpoint(results, direction)
    
    print_summary(results)


def run_arc_to_gsm8k():
    """Fine-tune on ARC, then run noise experiments on GSM8k."""
    direction = "arc_to_gsm8k"
    results, completed = load_checkpoint(direction)
    
    for model_id in BASE_MODELS:
        # Pre-train on ARC
        finetuned_path = pretrain_on_source(model_id, Dataset.ARC)
        
        # Run noise experiments on GSM8k
        for noise_level in NOISE_LEVELS:
            if (model_id, noise_level) in completed:
                print(f"Skipping {model_id} @ {noise_level:.0%} (done)")
                continue
            
            epoch_results = run_noise_experiment(
                finetuned_path, model_id, Dataset.ARC, Dataset.GSM8K, noise_level
            )
            for r in epoch_results:
                results.append(r.to_dict())
            completed.add((model_id, noise_level))
            save_checkpoint(results, direction)
    
    print_summary(results)


def print_summary(results: list[dict]):
    """Print experiment summary."""
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Source':<8} {'Target':<8} {'Noise':<8} {'Epoch':<6} {'Accuracy':<10}")
    print(f"{'-'*80}")
    for r in results:
        print(
            f"{r['model_label']:<25} {r['source_dataset']:<8} {r['target_dataset']:<8} "
            f"{r['noise_level']:<8.0%} {r['epoch']:<6} {r['accuracy']:<10.2%}"
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Transfer learning noise experiments")
    parser.add_argument(
        "--direction",
        choices=["gsm8k_to_arc", "arc_to_gsm8k", "both"],
        default="both",
        help="Which experiment direction to run",
    )
    args = parser.parse_args()
    
    torch.manual_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.direction in ("gsm8k_to_arc", "both"):
        run_gsm8k_to_arc()
    
    if args.direction in ("arc_to_gsm8k", "both"):
        run_arc_to_gsm8k()


if __name__ == "__main__":
    main()

