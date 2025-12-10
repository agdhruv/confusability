"""
ARC-Challenge Noise Experiment

Fine-tune models on ARC-Challenge with varying levels of label noise by swapping
the correct option to an incorrect one for a fraction of the training data.
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
from src.utils import load_model, load_tokenizer, to_lora

# ============== CONFIG ==============
# (model_id, use_chat_template) so each model can match eval prompt style.
MODELS: list[tuple[str, bool]] = [
    ("Qwen/Qwen2.5-0.5B-Instruct", True),
    ("Qwen/Qwen2.5-1.5B-Instruct", True),
    ("Qwen/Qwen2.5-3B-Instruct", True),
    ("Qwen/Qwen2.5-7B-Instruct", True),
    ("meta-llama/Llama-3.2-3B-Instruct", True),
    ("meta-llama/Llama-3.1-8B-Instruct", True),
    ("mistralai/Mistral-7B-Instruct-v0.3", True),
]
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_EPOCHS = 3
OUTPUT_DIR = Path("./results/arc_noise")
MAX_SEQ_LEN = 512
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EVAL_BATCH_SIZE = 128
EVAL_LIMIT = None  # e.g. 100 for debugging
_arc_path = Path(lm_eval.__file__).parent / "tasks" / "arc"
TASK_MANAGER = TaskManager(include_defaults=False, include_path=str(_arc_path))
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


def evaluate_arc(model, tokenizer, model_id: str, use_chat_template: bool) -> float:
    """Evaluate on ARC-Challenge."""
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    lm = InmemoryPeftLM(model, tokenizer, model_id, device=str(model.device), batch_size=EVAL_BATCH_SIZE)
    eval_task = "arc_challenge_chat" if use_chat_template else "arc_challenge"
    with torch.inference_mode():
        results = simple_evaluate( # type: ignore
            model=lm,
            task_manager=TASK_MANAGER,
            tasks=[eval_task],
            num_fewshot=0,
            batch_size=EVAL_BATCH_SIZE,
            verbosity="ERROR",
            limit=EVAL_LIMIT,
            cache_requests=True,
            gen_kwargs={"max_new_tokens": None},
            apply_chat_template=use_chat_template,
        )
    arc = results["results"][eval_task]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    # For chat tasks, the metric is "exact_match,remove_whitespace"
    # For base tasks, the metric is "acc_norm,none"
    if use_chat_template:
        return arc.get("exact_match,remove_whitespace", arc.get("exact_match,none", 0))
    else:
        return arc.get("acc_norm,none", arc.get("acc,none", 0))


def format_example(tokenizer, question: str, choice_texts: list[str], answer_letter: str, answer_text: str, use_chat_template: bool) -> str:
    """Format a single ARC example to mirror lm_eval prompts."""
    if use_chat_template:
        options = "\n".join(
            f"{chr(ord('A') + i)}. {choice}"
            for i, choice in enumerate(choice_texts)
        )
        prompt = (
            "Given the following question and four candidate answers (A, B, C and D), "
            "choose the best answer.\n"
            f"Question: {question}\n"
            f"{options}\n"
            'Your response should end with "The best answer is [the_answer_letter]" '
            "where the [the_answer_letter] is one of A, B, C or D."
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"The best answer is {answer_letter}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    prompt = f"Question: {question}\nAnswer:"
    return f"{prompt} {answer_text}"


def prepare_noised_dataset(tokenizer, noise_level: float, use_chat_template: bool, seed: int = 42):
    """Load ARC-Challenge and apply label noise by swapping the correct option."""
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
            texts.append(format_example(tokenizer, q, choice_texts, noisy_letter, noisy_text, use_chat_template))

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
    """Evaluate on ARC-Challenge at the end of each epoch."""
    
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
        acc = evaluate_arc(self.model, self.tokenizer, self.model_id, self.use_chat_template)
        print(f"Epoch {epoch} accuracy: {acc:.2%}")
        
        self.epoch_results.append((epoch, acc, loss))


def run_experiment(model_id: str, use_chat_template: bool, noise_level: float) -> list[ExperimentResult]:
    """Single experiment: train on noised data, evaluate after each epoch."""
    print(f"\n{'='*60}")
    print(f"{model_id.split('/')[-1]} | noise={noise_level:.0%}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(model_id)
    model = load_lora_model(model_id)
    if "instruct" in model_id.lower() and not use_chat_template:
        print("WARNING: Instruct model detected but chat template not enabled. Using default format.")
    train_dataset = prepare_noised_dataset(tokenizer, noise_level, use_chat_template)
    
    eval_callback = EpochEvalCallback(model, tokenizer, model_id, use_chat_template)
    
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

    for model_id, use_chat_template in MODELS:
        for noise_level in NOISE_LEVELS:
            if (model_id, noise_level) in completed:
                print(f"Skipping {model_id} @ {noise_level:.0%} (done)")
                continue
            
            epoch_results = run_experiment(model_id, use_chat_template, noise_level)
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
