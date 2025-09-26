import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["WANDB_PROJECT"] = "confusability"
import torch, math
from dataclasses import dataclass
from transformers import (AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from src.data import load_train_dataset, load_eval_dataset
from src.utils import load_tokenizer, load_model, to_lora
from src.evals import InmemoryPeftLM
from lm_eval import simple_evaluate

torch.random.manual_seed(42)

# MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_ID = "Qwen/Qwen2.5-3B"
MODEL_LABEL = MODEL_ID.split('/')[-1]
CURRICULUM = "random_labels" # in case we want to use another randomness scheme later
# Training data
STREAM_SPLIT = "train"  # streaming source for inputs
TRAIN_NAME = "allenai/c4"
TRAIN_CONFIG = "en"
PROCESSED_DATA_DIR = f"./data/c4_train_processed/{MODEL_LABEL}"
SEQ_LEN = 2048

# Evaluation data
EVAL_NAME = "wikitext"  # small clean eval
EVAL_CONFIG = "wikitext-103-raw-v1"
PROCESSED_EVAL_DATA_DIR = f"./data/wikitext_eval_processed/{MODEL_LABEL}"
TRAIN_NUM_BLOCKS = 10_000  # ~50k*2k ≈ 100M tokens capacity if consumed fully
EVAL_NUM_BLOCKS = 2000
EVAL_SEQ_LEN = 1024 # Use a smaller sequence length for evaluation to prevent OOMs

tokenizer = load_tokenizer(MODEL_ID)
train_ds = load_train_dataset(TRAIN_NAME, TRAIN_CONFIG, PROCESSED_DATA_DIR, tokenizer, SEQ_LEN, TRAIN_NUM_BLOCKS, STREAM_SPLIT)
eval_ds = load_eval_dataset(EVAL_NAME, EVAL_CONFIG, PROCESSED_EVAL_DATA_DIR, tokenizer, EVAL_SEQ_LEN, EVAL_NUM_BLOCKS)

# Print number of available tokens
print(f"Training tokens: {sum(len(f['input_ids']) for f in train_ds):,}") # type: ignore

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
eval_batch_size = 16

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
    per_device_eval_batch_size=eval_batch_size,
    prediction_loss_only=True,
    report_to="wandb"
)

# ---- Use clean eval collator and log perplexity without storing logits ----
class EvalCollatorTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        prev_collator = self.data_collator
        self.data_collator = eval_collator
        # get an example to check that the collator is working
        
        # run lm_eval
        device = str(model.device)
        lm_object = InmemoryPeftLM(model, tokenizer, MODEL_ID, device=device, batch_size=eval_batch_size)
        lm_results = simple_evaluate(
            model=lm_object,
            tasks=["hellaswag"],
            num_fewshot=0,
            batch_size=eval_batch_size,
            verbosity="ERROR", # avoid printing too much
            limit=2000,
            cache_requests=True,
        )
        assert lm_results is not None
        self.lm_results = lm_results['results'] # store for logging
        
        results = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        self.data_collator = prev_collator
        return results

    def log(self, logs, *args, **kwargs):
        is_eval_run = "eval_loss" in logs
        # Inject perplexity results before logs are emitted
        if is_eval_run:
            assert "eval_perplexity" not in logs, "eval_perplexity should not have been calculated yet"
            logs["eval_perplexity"] = math.exp(float(logs["eval_loss"]))

            assert hasattr(self, 'lm_results'), "lm_results should have been set in evaluate"
            # Inject lm_eval results before logs are emitted
            for task, scores in self.lm_results.items():
                logs[f"eval_{task}"] = scores # just logging everything
                # special handling for mmlu and hellaswag
                if task.startswith("mmlu"):
                    logs[f"eval_{task}_acc"] = scores.get('acc,none', None)
                elif task == "hellaswag":
                    logs[f"eval_{task}_acc_norm"] = scores.get('acc_norm,none', None)
                else:
                    raise ValueError(f"Unknown task: {task}")
            # to avoid stale references to old results
            del self.lm_results
        else:
            assert not any(key.startswith("eval_") for key in logs), "eval_ results not expected in training logs"
        return super().log(logs, *args, **kwargs)

trainer = EvalCollatorTrainer(
    model=model,
    args=args,
    train_dataset=train_ds, # type: ignore
    eval_dataset=eval_ds, # type: ignore
    data_collator=data_collator,
)

trainer.evaluate()
trainer.train()
