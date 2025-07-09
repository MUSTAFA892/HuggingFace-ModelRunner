
# 🧠 Imports
import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# 🚫 Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# 📄 Load your data
with open("data.json") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)

# 🔧 Format data
def format_sample(sample):
    return {
        "text": f"### Prompt:\n{sample['prompt']}\n\n### Response:\n{sample['response']}"
    }

dataset = dataset.map(format_sample)

# 🧠 Load tokenizer and model
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Required for padding
model = AutoModelForCausalLM.from_pretrained(model_id)

# ✅ Apply LoRA config (very lightweight)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # common for LLaMA
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # ✅ Only LoRA layers will be trainable

# 🧼 Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_function, batched=True)

# 🧠 Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ⚙️ Training config (CPU)
training_args = TrainingArguments(
    output_dir="./lora_llama",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"
)

# 🚀 Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 🔧 Train
trainer.train()

# 💾 Save LoRA adapters + tokenizer
model.save_pretrained("lora_llama")
tokenizer.save_pretrained("lora_llama")
