from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model (ensure you have access + token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token="your_token_here")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token="your_token_here")

# Add a padding token (reuse eos_token or add a new one)
tokenizer.pad_token = tokenizer.eos_token

prompt = "write me a code for linear search in python"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate text
generate_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=512,
    pad_token_id=tokenizer.pad_token_id
)

output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
print(output[0])
