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
import argparse

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lm_eval
import torch
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments

from src.evals import InmemoryPeftLM
from src.noise import systematic_offset, number_perturbation
from src.utils import load_model, load_tokenizer, to_lora

parser = argparse.ArgumentParser()
parser.add_argument("--clean_only", action="store_true", default=False)
parser.add_argument("--noise_strategy", type=str, required=True, choices=["systematic_offset", "random_perturbation"])
args = parser.parse_args()

# ============== CONFIG ==============
# (model_id, use_chat_template, revision)
MODELS = [
    ("meta-llama/Llama-3.1-8B", False, None),
    ("meta-llama/Llama-3.1-8B-Instruct", True, None),
]
NOISE_STRATEGY = args.noise_strategy
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLEAN_ONLY = args.clean_only  # Control experiment: train only on clean portion (remove noisy examples)
NUM_EPOCHS = 1
_suffix = "_clean_only" if CLEAN_ONLY else ""
OUTPUT_DIR = Path(f"./results/gsm8k_{NOISE_STRATEGY}{_suffix}")
MAX_SEQ_LEN = 512
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EVAL_BATCH_SIZE = 128
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


def evaluate_gsm8k(model, tokenizer, model_id: str, use_chat_template: bool) -> float:
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
            apply_chat_template=use_chat_template,
        )
    gsm8k = results["results"]["gsm8k"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return gsm8k.get("exact_match,strict-match", gsm8k.get("acc,none", 0))


def format_example(tokenizer, question: str, answer: str, use_chat_template: bool) -> str:
    """Format a single example, using chat template if enabled."""
    if use_chat_template:
        # Match lm_eval format to ensure evaluation is not out-of-distribution
        messages = [
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
            {"role": "assistant", "content": answer},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        return f"Question: {question}\nAnswer: {answer}"


def prepare_noised_dataset(tokenizer, noise_level: float, use_chat_template: bool, seed: int = 42):
    """Load GSM-8k and apply label noise."""
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    
    rng = random.Random(seed)
    n_noised = int(len(train_ds) * noise_level)
    noise_indices = set(rng.sample(range(len(train_ds)), n_noised)) if n_noised else set()

    # Control experiment: train only on clean portion
    if CLEAN_ONLY:
        clean_indices = [i for i in range(len(train_ds)) if i not in noise_indices]
        train_ds = train_ds.select(clean_indices)
        noise_indices = set()  # No noise to apply anymore
        print(f"CLEAN_ONLY: Training on {len(train_ds)} clean examples ({100*(1-noise_level):.0f}% of original)")

    def tokenize(examples, indices):
        texts = []
        prompts = []  # just the question part
        for idx, q, a in zip(indices, examples["question"], examples["answer"]):
            if idx in noise_indices:
                if NOISE_STRATEGY == "systematic_offset":
                    a = systematic_offset(a, offset=3.0)
                elif NOISE_STRATEGY == "random_perturbation":
                    a = number_perturbation(a, rng, scale=0.5)
                else:
                    raise ValueError(f"Invalid noise strategy: {NOISE_STRATEGY}")
            texts.append(format_example(tokenizer, q, a, use_chat_template))
            # Get prompt length for masking
            if use_chat_template:
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"Question: {q}\nAnswer:"}],
                    tokenize=False, add_generation_prompt=True
                ))
            else:
                prompts.append(f"Question: {q}\nAnswer: ")
        
        out = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        prompt_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        
        labels = []
        for ids, plen in zip(out["input_ids"], prompt_lens):
            lbl = [-100] * plen + list(ids[plen:])  # mask prompt tokens
            labels.append(lbl)
        out["labels"] = labels
        return out

    return train_ds.map(tokenize, batched=True, with_indices=True, remove_columns=train_ds.column_names)


def load_lora_model(model_id: str, revision: str | None = None):
    """Load model with LoRA adapter."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = load_model(model_id, dtype, revision=revision)
    print(f"Loaded model: {model_id} (rev={revision}) on device: {model.device}")
    return to_lora(model, checkpointing=True)


class EpochEvalCallback(TrainerCallback):
    """Evaluate on GSM8k at the end of each epoch."""
    
    def __init__(self, model, tokenizer, model_id: str, use_chat_template: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.use_chat_template = use_chat_template
        self.epoch_results: list[tuple[int, float, float]] = []  # (epoch, accuracy, loss)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        # Get loss from most recent log entry
        loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), 0.0)
        
        print(f"\n--- End of Epoch {epoch}/{NUM_EPOCHS} - Evaluating... ---")
        acc = evaluate_gsm8k(self.model, self.tokenizer, self.model_id, self.use_chat_template)
        print(f"Epoch {epoch} accuracy: {acc:.2%}")
        
        self.epoch_results.append((epoch, acc, loss))


def run_experiment(model_id: str, noise_level: float, use_chat_template: bool, revision: str | None = None) -> list[ExperimentResult]:
    """Single experiment: train on noised data, evaluate after each epoch."""
    # Create combined ID for tracking (includes revision suffix if present)
    full_id = f"{model_id}_{revision}" if revision else model_id
    label = full_id.split("/")[-1]
    
    print(f"\n{'='*60}")
    print(f"{label} | noise={noise_level:.0%}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(model_id, revision=revision)
    model = load_lora_model(model_id, revision=revision)
    train_dataset = prepare_noised_dataset(tokenizer, noise_level, use_chat_template)
    
    eval_callback = EpochEvalCallback(model, tokenizer, model_id, use_chat_template)
    
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"{label}_noise{noise_level:.2f}"),
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
            model_id=full_id,
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

    for model_id, use_chat_template, revision in MODELS:
        full_id = f"{model_id}_{revision}" if revision else model_id
        for noise_level in NOISE_LEVELS:
            if (full_id, noise_level) in completed:
                print(f"Skipping {full_id.split('/')[-1]} @ {noise_level:.0%} (done)")
                continue
            
            # Skip 100% noise in clean_only mode (no clean examples to train on)
            if CLEAN_ONLY and noise_level == 1.0:
                print(f"Skipping {full_id.split('/')[-1]} @ {noise_level:.0%} (no clean examples)")
                continue
            
            epoch_results = run_experiment(model_id, noise_level, use_chat_template, revision)
            for r in epoch_results:
                results.append(r.to_dict())
            completed.add((full_id, noise_level))
            save_checkpoint(results)

    # Final summary
    print(f"\n{'='*60}")
    print(f"{'Model':<45} {'Noise':<8} {'Epoch':<6} {'Accuracy':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['model_label']:<45} {r['noise_level']:<8.0%} {r['epoch']:<6} {r['post_train_accuracy']:<10.2%}")


if __name__ == "__main__":
    main()
