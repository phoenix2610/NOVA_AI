from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/tathya/models/minicpm5-1b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

messages = [{"role": "user", "content": "What is the Supertrend indicator and how is it used in scalping?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=300
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
