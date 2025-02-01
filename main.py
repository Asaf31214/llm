import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

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

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

threading.Thread(
    target=model.generate,
    kwargs={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 1000,
        "streamer": streamer,
    }
).start()

for token in streamer:
    print(token, end="", flush=True)
