from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import os
import torch
import threading
import json
import time

app = Flask(__name__)

# Load model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

# Context management
context_window = []  # Stores past messages and responses
max_context_tokens = 1500  # Reserve space for context to fit within 2048 tokens

# Path to training.json
training_file_path = "./luma-memory/training/training.json"

# Metadata file to track the number of new entries
metadata_file_path = "./luma-memory/training/training_metadata.json"

def load_training_data():
    """Load training data from the training.json file."""
    if not os.path.exists(training_file_path):
        print("No training data found.")
        return []

    with open(training_file_path, "r", encoding="utf-8") as file:
        training_data = json.load(file)
    
    return [(entry["user_input"], entry["ai_response"]) for entry in training_data]

def save_training_data(training_data):
    """Save updated training data back to the training.json file."""
    with open(training_file_path, "w", encoding="utf-8") as file:
        json.dump(training_data, file, ensure_ascii=False, indent=4)

def load_metadata():
    """Load metadata for training tracking."""
    if not os.path.exists(metadata_file_path):
        return {"new_entries": 0}
    with open(metadata_file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_metadata(metadata):
    """Save updated metadata."""
    with open(metadata_file_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)

@app.before_first_request
def check_training_data_on_startup():
    """Check if the number of training entries is more than 50 on app startup."""
    training_data = load_training_data()
    if len(training_data) > 50:
        print(f"Found {len(training_data)} entries in training.json. Starting fine-tuning...")
        start_fine_tuning()

@app.route("/add_training_data", methods=["POST"])
def add_training_data():
    """Add new training data and check if retraining is needed."""
    global model, tokenizer  # Ensure we access the global model and tokenizer

    # Load the current training data and metadata
    training_data = load_training_data()
    metadata = load_metadata()

    # Add new training data
    new_entry = request.json
    if not new_entry.get("user_input") or not new_entry.get("ai_response"):
        return jsonify({"error": "Both user_input and ai_response are required"}), 400
    
    training_data.append(new_entry)
    save_training_data(training_data)

    # Increment the new entries counter
    metadata["new_entries"] += 1
    save_metadata(metadata)

    # Check if retraining is needed
    if metadata["new_entries"] >= 50:
        print("50 new entries reached. Starting fine-tuning...")
        start_fine_tuning()
        metadata["new_entries"] = 0  # Reset the counter
        save_metadata(metadata)

    return jsonify({"message": "Training data added successfully!"})

def start_fine_tuning():
    """Run fine-tuning in a background thread."""
    print("Starting fine-tuning thread...")  # Log the start of fine-tuning
    threading.Thread(target=fine_tune_model, daemon=True).start()

def fine_tune_model():
    """Fine-tune the model based on the training data and save the updated model."""
    global model, tokenizer  # Ensure model and tokenizer are declared as global here
    print("Starting background fine-tuning...")

    # Load the training data
    training_data = load_training_data()
    if not training_data:
        print("No training data to fine-tune.")
        return

    # Prepare data for fine-tuning
    inputs = []
    labels = []
    for user_input, ai_response in training_data:
        inputs.append(user_input)
        labels.append(ai_response)

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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    try:
        print("Training in progress...")
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")

        # Reload the fine-tuned model into the app
        model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
        tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

        print("Fine-tuning complete. Updating model and clearing training data...")

        # Clear used training data
        save_training_data([])

    except Exception as e:
        print(f"Error during fine-tuning: {e}")

@app.route("/train_status", methods=["GET"])
def train_status():
    """Get the training status."""
    metadata = load_metadata()
    return jsonify({"new_entries": metadata.get("new_entries", 0)})

@app.route("/chat", methods=["POST"])
def chat():
    global context_window
    
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Append user input to context with separator
    context_window.append(f"User: {user_input}")

    # Ensure context fits within token limits
    while len(context_window) > 0 and sum(len(tokenizer(msg)['input_ids']) for msg in context_window) > max_context_tokens:
        context_window.pop(0)  # Remove oldest messages

    # Prepare the model input
    conversation_input = "\n".join(context_window) + "\nAI:"
    inputs = tokenizer(conversation_input, return_tensors="pt")

    # Generate a response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=inputs["input_ids"].size(1) + 150,  # Limit response length
        temperature=0.7,  # Control randomness
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.2,  # Penalize repeated phrases
        eos_token_id=tokenizer.eos_token_id  # End-of-sequence token
    )

    # Decode the generated text and clean up response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\nAI:")[-1].strip()

    # Trim response to avoid excessive length
    response = response[:500]  # Optional: limit the response length

    # Append AI response to context
    context_window.append(f"AI: {response}")

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)
