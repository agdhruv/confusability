"""
RLVR using GRPO (with SFT warmstart on GSM8K)

Phase 1: SFT on GSM8K to teach the #### format
Phase 2: GRPO to improve accuracy via RL
"""
import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

# ============== CONFIG ==============
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = Path("./results/rlvr_gsm8k")

# SFT Config
SFT_EPOCHS = 2
SFT_LR = 2e-5
SFT_BATCH_SIZE = 64
SFT_MICRO_BATCH = 64
SFT_MAX_LENGTH = 512

# GRPO Config
GRPO_EPOCHS = 1
GRPO_LR = 1e-5
NUM_GENERATIONS = 8
MAX_COMPLETION_LENGTH = 512
BETA = 0.01
# ====================================

SYSTEM_PROMPT = "You are a helpful math assistant. Solve the problem step by step."


def format_gsm8k_sft(example):
    """Format GSM8K for SFT training (prompt + completion)."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "completion": [
            {"role": "assistant", "content": example["answer"]},
        ],
    }


def format_gsm8k_grpo(example, tokenizer):
    """Format GSM8K for GRPO (prompt string only)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        "answer": example["answer"],
    }


def extract_answer(text):
    """Extract numerical answer from #### pattern."""
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def reward_accuracy(completions, prompts, answer, **kwargs):
    """Reward = 1.0 if correct, 0.0 otherwise."""
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        pred_val = extract_answer(completion)
        gt_val = extract_answer(ground_truth)
        if pred_val is not None and gt_val is not None and pred_val == gt_val:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def run_sft(tokenizer, dataset, output_dir: Path):
    """Phase 1: SFT on GSM8K to teach the #### format."""
    print(f"\n{'='*60}")
    print("Phase 1: SFT on GSM8K")
    print(f"{'='*60}")

    sft_dataset = dataset.map(format_gsm8k_sft, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    config = SFTConfig(
        output_dir=str(output_dir / "sft"),
        per_device_train_batch_size=SFT_MICRO_BATCH,
        gradient_accumulation_steps=SFT_BATCH_SIZE // SFT_MICRO_BATCH,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LR,
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=SFT_MAX_LENGTH,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        args=config,
    )
    trainer.train()

    save_path = output_dir / "sft_model"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"SFT model saved to {save_path}")

    del model, trainer
    torch.cuda.empty_cache()
    return save_path


def run_grpo(tokenizer, dataset, sft_model_path: Path, output_dir: Path):
    """Phase 2: GRPO to improve accuracy via RL."""
    print(f"\n{'='*60}")
    print("Phase 2: GRPO on GSM8K")
    print(f"{'='*60}")

    grpo_dataset = dataset.map(
        lambda x: format_gsm8k_grpo(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # EOS tokens for Qwen
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_ids = [tokenizer.eos_token_id, im_end_token_id]

    config = GRPOConfig(
        output_dir=str(output_dir / "grpo"),
        run_name="grpo_gsm8k",
        learning_rate=GRPO_LR,
        num_train_epochs=GRPO_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        beta=BETA,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to="wandb",
        use_vllm=False,
        temperature=0.7,
        log_completions=True,
        gradient_checkpointing=True,
        generation_kwargs={"eos_token_id": eos_token_ids},
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_accuracy,
        args=config,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    save_path = output_dir / "grpo_model"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"GRPO model saved to {save_path}")


def main():
    torch.manual_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main", split="train")

    sft_model_path = OUTPUT_DIR / "sft_model"
    grpo_model_path = OUTPUT_DIR / "grpo_model"

    # Phase 1: SFT
    if sft_model_path.exists():
        print(f"SFT model already exists at {sft_model_path}, skipping SFT...")
    else:
        run_sft(tokenizer, dataset, OUTPUT_DIR)

    # Phase 2: GRPO
    if grpo_model_path.exists():
        print(f"GRPO model already exists at {grpo_model_path}, skipping GRPO...")
    else:
        run_grpo(tokenizer, dataset, sft_model_path, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"SFT model: {sft_model_path}")
    print(f"GRPO model: {grpo_model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
