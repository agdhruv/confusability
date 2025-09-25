import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from dataclasses import dataclass
from datasets import load_dataset, Dataset, load_from_disk
from itertools import islice
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model

torch.random.manual_seed(42)

MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_LABEL = MODEL_ID.split('/')[-1]
SEQ_LEN = 2048
EVAL_SEQ_LEN = 1024 # Use a smaller sequence length for evaluation to prevent OOMs
STREAM_SPLIT = "train"  # streaming source for inputs
EVAL_NAME = "wikitext"  # small clean eval
EVAL_CONFIG = "wikitext-103-raw-v1"
PROCESSED_DATA_DIR = f"./c4_train_processed/{MODEL_LABEL}"
MATSZ = 10_000  # ~50k*2k ≈ 100M tokens capacity if consumed fully
EVAL_MATSZ = 2000

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- Streaming text -> fixed-length blocks (IterableDataset style) ----
def chunk_stream(ds, block_size=SEQ_LEN):
    buf = []
    for ex in ds:
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids: 
            continue
        buf.extend(ids)
        while len(buf) >= block_size:
            yield {"input_ids": buf[:block_size]}
            buf = buf[block_size:]

# Build a small materialized buffer so Trainer can sample batches easily.
# (Enough for a few thousand steps; we don't need the whole C4.)
if not os.path.exists(PROCESSED_DATA_DIR):
    print(f"Processed data not found. Creating and saving to {PROCESSED_DATA_DIR}...")
    train_stream = load_dataset("allenai/c4", "en", split=STREAM_SPLIT, streaming=True)
    train_iterable = chunk_stream(train_stream)
    materialized = list(islice(train_iterable, MATSZ))
    train_ds = Dataset.from_list(materialized)
    train_ds.save_to_disk(PROCESSED_DATA_DIR)
else:
    print(f"Loading processed data from {PROCESSED_DATA_DIR}...")
    train_ds = load_from_disk(PROCESSED_DATA_DIR)

# Print number of available tokens
print(f"Number of available tokens: {sum(len(f['input_ids']) for f in train_ds):,}")

# Clean eval (no noise) for perplexity
eval_raw = load_dataset(EVAL_NAME, EVAL_CONFIG, split="test")
eval_iterable = chunk_stream(eval_raw, block_size=EVAL_SEQ_LEN)
eval_ds: Dataset = Dataset.from_list(list(islice(eval_iterable, EVAL_MATSZ)))

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

# TODO: probably remove, doesn't even make sense
@dataclass
class ShuffleTokenCollator:
    def __call__(self, features):
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        B, L = input_ids.size()
        # shuffle tokens independently per sample
        idx = torch.stack([torch.randperm(L) for _ in range(B)])
        shuffled = input_ids.gather(1, idx)
        # causal LM expects labels==input_ids (shift happens inside the model)
        labels = shuffled.clone()
        attention_mask = torch.ones_like(shuffled)
        return {"input_ids": shuffled, "labels": labels, "attention_mask": attention_mask}

# Choose curriculum via ENV var
CURRICULUM = os.environ.get("CURRICULUM", "random_labels")  # or "shuffle"
if CURRICULUM == "shuffle":
    data_collator = ShuffleTokenCollator()
else:
    data_collator = RandomLabelCollator(tokenizer, tokenizer.vocab_size)

# Clean eval collator
eval_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---- Model + LoRA ----
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch_dtype, device_map="auto"
)
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, lora)
# Ensure compatibility with gradient checkpointing
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
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

trainer.save_model(os.path.join(args.output_dir, "final"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))