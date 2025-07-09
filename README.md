# 🤖 HuggingFace-ModelRunner

A modular project to **run**, **fine-tune**, and **test** various Hugging Face models — including **Meta’s LLaMA**, **LoRA fine-tuned variants**, and others. This repository is designed for **lightweight local testing**, **personalized training**, and **experimentation** with different model architectures and generation strategies.

---

## 🔑 Prerequisites

Before using any of the scripts, follow these steps:

### 1️⃣ Log in to Hugging Face CLI

Make sure you are authenticated to download models:

```bash
huggingface-cli login
```

> If you don’t have the CLI installed:

```bash
pip install huggingface_hub
```

---

### 2️⃣ Create Virtual Environment (Recommended)

We suggest isolating your environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### 3️⃣ Install All Dependencies

Install all the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> This will install:

* `transformers`
* `datasets`
* `torch`
* `accelerate`
* `peft`
* `sentencepiece`
  and other essential tools.

---

## 🚀 How to Use

Each subfolder contains a **README** with exact instructions, but generally:

* ✅ `Llama/`: Run the LLaMA 1B model with different text generation methods.
* ✅ `fine_tuned_llm/`: Fine-tune a model using LoRA adapters and test custom responses.

---

## 🧠 Use Cases

* Personal assistant LLMs (like MustafaGPT)
* Instruction-following models
* Lightweight fine-tuning on local CPU/GPU/TPU
* Educational and research experiments

---

## 📌 Notes

* Fine-tuning is optimized for **low-resource environments**
* All output models are saved locally (`./lora_llama`, `./fine_tuned_llama`, etc.)
* You can upload the fine-tuned adapters or full models to [huggingface.co](https://huggingface.co/) if desired

---

## 🤝 Contributing

Pull requests are welcome if you want to add support for other model types, datasets, or training strategies.

---

## 🧾 License

MIT License. Free to use and modify.
