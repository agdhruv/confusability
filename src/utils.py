from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model


def load_tokenizer(model_id: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_id: str, torch_dtype: torch.dtype, revision: str | None = None):
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, device_map="auto", use_safetensors=True, revision=revision
    )
    return model

def to_lora(model, checkpointing: bool):
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora)
    if checkpointing:
        # Ensure compatibility with gradient checkpointing
        model.config.use_cache = False # type: ignore
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads() # type: ignore
    return model