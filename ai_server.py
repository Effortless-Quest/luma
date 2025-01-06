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

def load_training_data():
    training_file_path = "./luma-memory/training/training.md"
    if not os.path.exists(training_file_path):
        print("No training data found.")
        return []

    training_data = []
    with open(training_file_path, "r", encoding="utf-8") as file:
        entries = file.read().split("\n---\n")
        for entry in entries:
            if entry.strip():
                parts = entry.split("### AI Response:")
                user_input = parts[0].replace("### User Input:", "").strip()
                ai_response = parts[1].strip() if len(parts) > 1 else ""
                training_data.append((user_input, ai_response))
    return training_data

def update_memory(user_input, response):
    """Update the memory with new user input and model response."""
    global memory

    memory_entry = {
        "user_input": user_input,
        "response": response,
    }
    memory_key = f"interaction_{len(memory)}"  # Unique key for each memory
    memory[memory_key] = memory_entry
    
    # Save the memory to the file
    with open(memory_file, "w") as file:
        json.dump(memory, file, indent=4)

    # Fine-tune the model immediately after updating memory
    start_fine_tuning()

def start_fine_tuning():
    """Run fine-tuning in a background thread."""
    threading.Thread(target=fine_tune_model).start()

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
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
    label_ids = tokenizer(labels, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids

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
        per_device_train_batch_size=1,  # Reduce batch size if you're encountering memory issues
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=2,
        overwrite_output_dir=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Add a custom callback to monitor training progress
    def log_training_progress(args, state, control, logs=None):
        if logs is not None:
            print(f"Training Step {state.global_step} - Loss: {logs.get('loss', 'N/A')}")
        return control

    trainer.add_callback(log_training_progress)

    # Fine-tune the model
    print("Training in progress...")
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

@app.route("/edit_response", methods=["POST"])
def edit_response():
    global memory
    
    data = request.json
    user_input = data.get("user_input")
    ai_response = data.get("ai_response")
    
    if not user_input or not ai_response:
        return jsonify({"error": "Both user_input and ai_response are required"}), 400

    # Generate a unique key for this new interaction
    interaction_key = f"interaction_{len(memory)}"
    
    # Save this interaction in the memory
    memory[interaction_key] = {
        "user_input": user_input,
        "response": ai_response
    }
    
    # Save the memory to the file
    with open(memory_file, "w") as file:
        json.dump(memory, file, indent=4)
    
    # Fine-tune the model after this update
    start_fine_tuning()
    
    return jsonify({"message": "Training data updated successfully"})

if __name__ == "__main__":
    # Load and process training data before starting the app
    training_data = load_training_data()
    for user_input, ai_response in training_data:
        update_memory(user_input, ai_response)  # Populate memory with pre-existing training data
    
    app.run(port=5000)
