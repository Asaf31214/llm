import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token =tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder="offload"
)

input_text = "Hello, how are you?"
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
        "top_p": 0.9,
        "top_k": 50,
        "temperature": 0.7,
    }
).start()

for token in streamer:
    print(token, end="", flush=True)
