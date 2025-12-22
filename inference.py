import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "./results/alpaca_soup/Qwen2.5-0.5B/ingredient_0"
MODEL_ID = "./results/dpo_experiment/dpo_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)

prompt = tokenizer.apply_chat_template([
    {"role": "system", "content": "You are a helpful math assistant. Please reason step by step and put your final answer after #### (e.g. #### 12)"},
    {"role": "user", "content": "Marie has 98 unread messages on her phone. She decides to clear them by reading 20 messages a day. However, she also gets 6 new messages a day. How many days will it take her to read all her unread messages?"}
], return_tensors="pt", add_generation_prompt=True).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|im_end|>")
]
outputs = model.generate(prompt, max_new_tokens=128, eos_token_id=terminators)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))