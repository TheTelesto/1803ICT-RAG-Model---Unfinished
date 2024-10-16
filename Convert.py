from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained('./fine_tuned_model')

# Save the model in GGUF format
model.save_pretrained('./fine_tuned_model_gguf', format='gguf')