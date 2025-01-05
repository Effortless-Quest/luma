from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json
import os
import torch
import threading

app = Flask(__name__)

# Load model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

# Context management
context_window = []  # Stores past messages and responses
max_context_tokens = 1500  # Reserve space for context to fit within 2048 tokens

# Memory storage (JSON file-based simulation)
memory_file = "memory.json"
if os.path.exists(memory_file):
    with open(memory_file, "r") as file:
        memory = json.load(file)
else:
    memory = {}

interaction_count = 0  # Track number of interactions

def update_memory(user_input, response):
    """Update the memory with new user input and model response."""
    global memory, interaction_count

    memory_entry = {
        "user_input": user_input,
        "response": response,
    }
    memory_key = f"interaction_{len(memory)}"  # Unique key for each memory
    memory[memory_key] = memory_entry
    
    # Save the memory to the file
    with open(memory_file, "w") as file:
        json.dump(memory, file, indent=4)

    # Increment interaction count
    interaction_count += 1

    # Trigger fine-tuning every 50 interactions
    if interaction_count >= 50:
        interaction_count = 0
        threading.Thread(target=fine_tune_model, daemon=True).start()  # Start fine-tuning in the background

def fine_tune_model():
    """Fine-tune the model based on collected memories and save the updated model."""
    global model, tokenizer  # Ensure model and tokenizer are declared as global here
    print("Starting background fine-tuning...")

    # Prepare data for fine-tuning
    inputs = []
    labels = []
    for entry in memory.values():
        inputs.append(entry["user_input"])
        labels.append(entry["response"])

    # Tokenize inputs and labels
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids
    label_ids = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids

    # Create a Dataset (a simple torch Dataset in this case)
    class MemoryDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, label_ids):
            self.input_ids = input_ids
            self.label_ids = label_ids

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "labels": self.label_ids[idx],
            }

        def __len__(self):
            return len(self.input_ids)

    dataset = MemoryDataset(input_ids, label_ids)

    # Fine-tuning arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=1,  # For simplicity, fine-tuning for 1 epoch (can be adjusted)
        per_device_train_batch_size=2,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        overwrite_output_dir=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    # Reload the fine-tuned model into the app
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

    print("Fine-tuning complete and base model updated!")

@app.route("/chat", methods=["POST"])
def chat():
    global context_window
    
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Tokenize user input
    input_tokens = tokenizer(user_input, return_tensors="pt")["input_ids"]
    input_token_count = input_tokens.size(1)

    # Truncate context if needed
    while len(context_window) > 0 and input_token_count + sum(len(tokenizer(msg)['input_ids']) for msg in context_window) > max_context_tokens:
        context_window.pop(0)  # Remove oldest messages to fit within token limit

    # Add user input to context
    context_window.append(user_input)

    # Retrieve relevant memories and integrate them into context
    memory_integration = " ".join([entry['response'] for entry in memory.values()])
    conversation_input = memory_integration + " " + " ".join(context_window)

    # Prepare full conversation input
    inputs = tokenizer(conversation_input, return_tensors="pt")

    # Generate a response
    outputs = model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add the model's response to the context
    context_window.append(response)

    # Update memory with the new interaction
    update_memory(user_input, response)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)
