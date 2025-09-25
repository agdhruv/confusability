import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from dataclasses import dataclass
from transformers import (AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from src.data import load_data_with_cache, load_eval_dataset
from src.utils import load_tokenizer, load_model, to_lora

torch.random.manual_seed(42)

MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_LABEL = MODEL_ID.split('/')[-1]
SEQ_LEN = 2048
EVAL_SEQ_LEN = 1024 # Use a smaller sequence length for evaluation to prevent OOMs
STREAM_SPLIT = "train"  # streaming source for inputs
EVAL_NAME = "wikitext"  # small clean eval
EVAL_CONFIG = "wikitext-103-raw-v1"
PROCESSED_DATA_DIR = f"./data/c4_train_processed/{MODEL_LABEL}"
MATSZ = 10_000  # ~50k*2k ≈ 100M tokens capacity if consumed fully
EVAL_MATSZ = 2000
CURRICULUM = "random_labels" # in case we want to use another randomness scheme later

tokenizer = load_tokenizer(MODEL_ID)
train_ds = load_data_with_cache(PROCESSED_DATA_DIR, SEQ_LEN, tokenizer, MATSZ, STREAM_SPLIT)
eval_ds = load_eval_dataset(EVAL_NAME, EVAL_CONFIG, tokenizer, EVAL_SEQ_LEN, EVAL_MATSZ)

# Print number of available tokens
print(f"Number of available tokens: {sum(len(f['input_ids']) for f in train_ds):,}")

@dataclass
class RandomLabelCollator:
    tokenizer: AutoTokenizer
    vocab_size: int
    dynamic: bool = True
    def __call__(self, features):
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        # labels are fresh random tokens each batch (maximally confusing)
        labels = torch.randint(low=0, high=self.vocab_size, size=input_ids.shape, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

data_collator = RandomLabelCollator(tokenizer, tokenizer.vocab_size)

# Clean eval collator for perplexity on clean data
eval_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---- Model + LoRA ----
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
model = load_model(MODEL_ID, torch_dtype)
model = to_lora(model, checkpointing=True)
model.print_trainable_parameters()

# ---- Hyperparameters ----
batch_size = 64
micro_batch_size = 16
assert batch_size % micro_batch_size == 0, "batch_size must be divisible by micro_batch_size"
grad_accum = batch_size // micro_batch_size
num_tokens_per_step = batch_size * SEQ_LEN

# ---- Training args: tiny, fast ----
args = TrainingArguments(
    output_dir=f"runs/{MODEL_LABEL}/{CURRICULUM}",
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=0,
    logging_steps=1,
    max_steps=400,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    lr_scheduler_type="cosine",
    save_steps=100,
    save_safetensors=True,
    eval_steps=10,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    prediction_loss_only=True,
    report_to="wandb"
)

# ---- Use clean eval collator and log perplexity without storing logits ----
class EvalCollatorTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self.data_collator = eval_collator
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def log(self, logs, *args, **kwargs):
        # Inject perplexity before logs are emitted
        if logs and "eval_loss" in logs and "eval_perplexity" not in logs:
            import math
            logs["eval_perplexity"] = math.exp(float(logs["eval_loss"]))
        return super().log(logs, *args, **kwargs)

trainer = EvalCollatorTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

trainer.evaluate()
trainer.train()
