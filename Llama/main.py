import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    device = -1
)

print(pipe("can u write me the code for linear search in python"))
