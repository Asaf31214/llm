import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    torch_dtype=torch.float16,
)

model = model.to("cuda")

input_text = "Hello, does 2 + 2 equal 5? Give certain answer."
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")
outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10000)
for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))
