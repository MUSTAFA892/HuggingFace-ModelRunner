# test.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 🔧 Load base model + tokenizer
base_model_id = "meta-llama/Llama-3.2-1B"
lora_path = "lora_llama"

# Load tokenizer from LoRA folder
tokenizer = AutoTokenizer.from_pretrained(lora_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is handled

# Load base model and apply LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, lora_path)

model.eval()  # Inference mode

# 📝 Input prompt
user_input = "What is your name?"
prompt = f"### Prompt:\n{user_input}\n\n### Response:\n"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# 🔮 Generate response
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# 📤 Decode and clean output
full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = full_output.split("### Response:")[-1].strip()

print("\n🤖 Response:")
print(response)
