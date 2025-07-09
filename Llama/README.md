## 🦙 LLaMA 3.2 1B – Simple Text Generation with Hugging Face

This script demonstrates a minimal setup for generating text using the [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) model from Hugging Face Transformers.

---

### 📌 What It Does

* Loads the **LLaMA 3.2 1B** model via Hugging Face's `pipeline` API
* Uses `text-generation` to respond to a natural language prompt
* Runs on **CPU** (`device=-1`)
* Prints the model’s generated response

---

### 🧪 Example Prompt

```text
"can u write me the code for linear search in python"
```

### ✅ Expected Output

The model will generate a Python function for linear search, for example:

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

---

### ⚙️ Requirements

Install the required Python packages:

```bash
pip install transformers
```

You don't need to manually download the model — Hugging Face will do it automatically on first run.

---

### ▶️ Run the Script

```bash
python script.py
```

---

### 💡 Notes

* This runs entirely on CPU. For GPU support, change `device=-1` to `device=0`.
* This is a basic use case. You can customize the prompt or integrate it into larger apps.