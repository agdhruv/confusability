"""
Alpaca Model Soup Training

Train multiple full fine-tuned "ingredients" with different seeds, then average weights.
"""
import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ============== CONFIG ==============
MODEL_ID = "Qwen/Qwen2.5-0.5B"
NUM_INGREDIENTS = 3
NUM_EPOCHS = 3
OUTPUT_DIR = Path("./results/alpaca_soup")
MAX_LENGTH = 1024
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
# ====================================


def format_alpaca(example):
    """Format single Alpaca example as prompt-completion."""
    if example["input"]:
        user_content = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_content = example['instruction']
    
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [{"role": "assistant", "content": example['output']}],
    }


def load_model():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="auto")


def train_ingredient(tokenizer, dataset, seed_idx: int, output_dir: Path) -> Path:
    """Train single ingredient, save and return path."""
    seed = 42 + seed_idx
    print(f"\n{'='*60}")
    print(f"Training Ingredient {seed_idx + 1}/{NUM_INGREDIENTS} (seed={seed})")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    random.seed(seed)

    model = load_model()
    save_path = output_dir / f"ingredient_{seed_idx}"

    config = SFTConfig(
        output_dir=str(save_path),
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
        seed=seed,
        data_seed=seed,
        max_length=MAX_LENGTH,
        completion_only_loss=True,  # Only compute loss on completion (assistant response)
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=config,
    )
    trainer.train()
    trainer.save_model(str(save_path))

    del model, trainer
    torch.cuda.empty_cache()
    return save_path


def make_soup(ingredient_paths: list[Path], output_dir: Path):
    """Load ingredients, average weights, save soup."""
    print(f"\n{'='*60}")
    print(f"Making Soup from {len(ingredient_paths)} ingredients")
    print(f"{'='*60}")

    soup_state = None
    for path in ingredient_paths:
        print(f"Loading {path.name}...")
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")
        state = model.state_dict()

        if soup_state is None:
            soup_state = {k: v.clone() for k, v in state.items()}
        else:
            for k in soup_state:
                soup_state[k] += state[k]
        del model

    assert soup_state is not None
    for k in soup_state:
        soup_state[k] /= len(ingredient_paths)

    print("Saving soup...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")
    model.load_state_dict(soup_state)
    model.save_pretrained(str(output_dir / "soup"))
    del model
    torch.cuda.empty_cache()


def main():
    model_label = MODEL_ID.split("/")[-1]
    output_dir = OUTPUT_DIR / model_label
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.save_pretrained(str(output_dir))

    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:20000]")
    dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)  # type: ignore

    # Train ingredients
    ingredient_paths = []
    for i in range(NUM_INGREDIENTS):
        path = train_ingredient(tokenizer, dataset, i, output_dir)
        ingredient_paths.append(path)

    # Make soup
    make_soup(ingredient_paths, output_dir)

    print(f"\n{'='*60}")
    print(f"Done! Saved {NUM_INGREDIENTS} ingredients + soup to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
