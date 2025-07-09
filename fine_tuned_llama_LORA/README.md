# 🦙 Fine-Tuning LLaMA 3.2 1B with LoRA (Low-Rank Adaptation)

This project demonstrates how to **fine-tune Meta's LLaMA 3.2 1B** model using **LoRA** (Low-Rank Adaptation) for efficient training and deployment — even on low-resource hardware like CPUs or minimal GPUs.

The repo includes:

* `train.py`: Fine-tuning the base LLaMA model with lightweight LoRA adapters using custom instruction-style data.
* `test.py`: Loading the fine-tuned model and generating responses from new prompts.

---

## 📌 Key Highlights

* ✅ Efficient training using **LoRA adapters** (only a small subset of model weights are trained).
* ✅ Runs on **CPU or low-memory GPU** (1B model).
* ✅ Uses **Hugging Face Transformers** + **PEFT** library.
* ✅ Simple input format: Prompt → Response.
* ✅ Saves **only the LoRA weights**, not the entire base model.
* ✅ Easy to test and deploy with `test.py`.

---

## 🧠 How It Works

### 🔧 1. Dataset Format

You provide a `data.json` file with this structure:

```json
[
  {
    "prompt": "What is your name?",
    "response": "My name is MustafaGPT."
  },
  ...
]
```

Each pair is converted internally into a single instruction-style string with a clear delimiter between **Prompt** and **Response**.

---

### 📦 2. Tokenization & Formatting

Each sample is formatted like:

```
### Prompt:
<your_prompt>

### Response:
<your_response>
```

The tokenizer handles padding, truncation, and max length to prepare inputs for training.

---

### 🏗️ 3. LoRA Configuration

We apply a LoRA config targeting only `q_proj` and `v_proj` (standard for LLaMA) with:

* `r = 8`
* `alpha = 16`
* `dropout = 0.1`
* `bias = none`

This ensures **minimal training overhead** while still letting the model learn new behavior from the dataset.

---

### 🏋️ 4. Training Setup

* **Model**: `meta-llama/Llama-3.2-1B`
* **Batch Size**: 1 (to fit on CPU)
* **Epochs**: 3
* **LoRA Parameters Only**: Trains just the injected adapter layers.
* **Trainer**: Uses Hugging Face’s `Trainer` API with a Causal Language Modeling data collator.

The model is saved using `model.save_pretrained()` — this saves only the LoRA adapter weights in the `lora_llama/` folder.

---

### 🧪 5. Inference (Testing)

`test.py` loads the original base model and injects the trained LoRA adapters. It constructs a new prompt (same format as training), generates a response using:

* **Top-p (nucleus) sampling**
* **Temperature** control for diversity

The output is then decoded and cleaned to extract just the meaningful reply (everything after `### Response:`).

---

## 🚀 How to Use

### 1. Prepare Your Data

Create a `data.json` file with `prompt` and `response` keys.

### 2. Install Dependencies

```bash
pip install torch transformers datasets peft accelerate sentencepiece
```

> You can also use `requirements.txt` if provided.

### 3. Run Fine-Tuning

```bash
python train.py
```

This will save the trained LoRA adapters to `./lora_llama`.

### 4. Run Inference

```bash
python test.py
```

You’ll be prompted to enter a question, and the model will generate a learned response.

---

## 📁 Output Structure

* `lora_llama/`: Contains saved LoRA weights and tokenizer config
* `logs/`: Contains training logs (if enabled)
* `data.json`: Your custom dataset

---

## ⚙️ Customization Tips

* Want better results? Increase dataset size, epochs, or tweak `top_p`, `temperature`.
* Want faster training? Use `gradient_checkpointing`, lower `max_length`, or `bf16` on GPUs.
* Want GPU/TPU? Just set `device_map="auto"` and use Accelerate.

---

## 🧠 Ideal Use Cases

* Custom Q\&A bots
* Personal assistants (e.g., MustafaGPT)
* Embedding structured knowledge into small models
* Lightweight inference deployments

---

## 📎 References

* [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
* [Transformers Text Generation](https://huggingface.co/docs/transformers/en/task_summary#text-generation)
* [LLaMA Model on Hugging Face](https://huggingface.co/meta-llama)