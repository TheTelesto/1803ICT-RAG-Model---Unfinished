import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# Step 1: Load your dataset
dataset = load_dataset('json', data_files='dataset.json')

# Step 2: Split the dataset into training and evaluation sets
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Step 3: Load the smaller model and tokenizer
model_name = "gpt2"  # You can use "distilgpt2" or "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Step 4: Apply LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=8,                     # Low-rank adaptation size
    lora_alpha=32,            # Scaling factor
    target_modules=["c_attn", "c_proj"],  # Target certain layers (attention)
    lora_dropout=0.1,         # LoRA dropout
    bias="none",              # Whether to train biases
    task_type="CAUSAL_LM"     # Causal language modeling task
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Step 5: Preprocess your dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Step 6: Set up training arguments with GPU and LoRA optimizations
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,   # Adjust batch size if necessary
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    fp16=True,  # Enable mixed precision training for efficiency
    gradient_accumulation_steps=4,  # Use gradient accumulation if memory is tight
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,  # Log every 10 steps
    report_to="tensorboard",  # Enable TensorBoard logging
)

# Step 7: Initialize the CustomTrainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        
        # Extract the logits
        logits = outputs.logits
        
        # Compute the loss
        labels = inputs["input_ids"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # Provide the evaluation dataset
)

# Step 8: Fine-tune the model
trainer.train()

# Step 9: Evaluate the model (optional)
trainer.evaluate()

# Step 10: Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Step 11: Test the fine-tuned model with an input prompt
input_text = "What is a functional requirement?"
inputs = tokenizer(input_text, return_tensors="pt")

# Ensure inputs are on the correct device
inputs = {key: value.to(model.device) for key, value in inputs.items()}

outputs = model.generate(inputs['input_ids'], max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))