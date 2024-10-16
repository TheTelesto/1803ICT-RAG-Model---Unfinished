from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the fine-tuned model and tokenizer
model_directory = './fine_tuned_model'
model = AutoModelForCausalLM.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Ensure the pad_token_id is set correctly
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Step 2: Define a function to ask questions and get responses
def ask_question(question, model, tokenizer, max_length=1024):
    # Prepare the input text
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    
    # Ensure inputs are on the correct device
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Generate the model's response
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        attention_mask=inputs['attention_mask'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode and display the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 3: Ask questions and get responses
questions = [
    "Explain what a functional requirement is and why it is important in software development. Provide examples to illustrate your points."
]

for question in questions:
    response = ask_question(question, model, tokenizer)
    print(f"Question: {question}")
    print(f"Response: {response}\n")