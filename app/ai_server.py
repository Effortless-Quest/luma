from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import os
import hashlib
import torch
import threading
import json

app = Flask(__name__)

# Directories and file paths
TRAINING_FILE = "./luma-memory/training/training.json"
MODEL_DIR = "./fine_tuned_model"
BASE_MODEL_NAME = "bigscience/bloomz-560m"
HASH_FILE_PATH = "./training_hash.txt"
max_context_tokens = 1500

# Load model (fine-tuned or base)
if os.path.exists(MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
else:
    # If there's no fine-tuned model, start with base model and save it as the first pre-trained model
    print("No fine-tuned model found, loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    model.save_pretrained(MODEL_DIR)  # Save the base model as the first pre-trained model
    tokenizer.save_pretrained(MODEL_DIR)  # Save the tokenizer
    print(f"Base model saved to {MODEL_DIR}")

context_window = []

def hash_file(file_path):
    """Generate a hash for the training file to detect changes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_training_data():
    """Load training data from training.json."""
    if not os.path.exists(TRAINING_FILE):
        print(f"Training data file not found at {TRAINING_FILE}. Creating an empty file.")
        with open(TRAINING_FILE, 'w', encoding='utf-8') as file:
            json.dump([], file)  # Create an empty file if it doesn't exist
        return []
    with open(TRAINING_FILE, "r", encoding="utf-8") as file:
        return [{"user_input": entry["user_input"], "ai_response": entry["ai_response"]} for entry in json.load(file)]

def fine_tune_model():
    """Fine-tune the model using the provided training data."""
    global model, tokenizer
    training_data = load_training_data()
    if not training_data:
        print("No training data to fine-tune.")
        return

    # Prepare inputs and labels
    inputs, labels = zip(*[(d["user_input"], d["ai_response"]) for d in training_data])
    input_ids = tokenizer(list(inputs), return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
    label_ids = tokenizer(list(labels), return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids

    class MemoryDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, label_ids):
            self.input_ids = input_ids
            self.label_ids = label_ids

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "labels": self.label_ids[idx]}

        def __len__(self):
            return len(self.input_ids)

    dataset = MemoryDataset(input_ids, label_ids)
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=100,
        save_total_limit=1,
        overwrite_output_dir=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    try:
        trainer.train()
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        # Reload fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("Fine-tuning complete. Model updated.")

    except Exception as e:
        print(f"Error during fine-tuning: {e}")

def check_and_train():
    """Check for new training data and fine-tune the model if needed."""
    if os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, "r") as f:
            previous_hash = f.read().strip()
    else:
        previous_hash = ""

    current_hash = hash_file(TRAINING_FILE)
    print(f"Previous Hash: {previous_hash}")
    print(f"Current Hash: {current_hash}")

    if current_hash != previous_hash:
        print("New training data detected. Fine-tuning the model...")
        fine_tune_model()

        with open(HASH_FILE_PATH, "w") as f:
            f.write(current_hash)
    else:
        print("No new training data found.")

@app.route("/add_training_data", methods=["POST"])
def add_training_data():
    training_data = load_training_data()
    new_entry = request.json
    if not new_entry.get("user_input") or not new_entry.get("ai_response"):
        return jsonify({"error": "Invalid training data"}), 400
    training_data.append(new_entry)
    with open(TRAINING_FILE, "w", encoding="utf-8") as file:
        json.dump(training_data, file, indent=4)

    threading.Thread(target=check_and_train, daemon=True).start()
    return jsonify({"message": "Training data added successfully!"})

@app.route("/chat", methods=["POST"])
def chat():
    global context_window
    
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    context_window.append(f"User: {user_input}")
    while len(context_window) > 0 and sum(len(tokenizer(msg)['input_ids']) for msg in context_window) > max_context_tokens:
        context_window.pop(0)

    conversation_input = "\n".join(context_window) + "\nAI:"
    inputs = tokenizer(conversation_input, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=inputs["input_ids"].size(1) + 150,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\nAI:")[-1].strip()
    response = response[:500]
    context_window.append(f"AI: {response}")

    return jsonify({"response": response})

if __name__ == "__main__":
    check_and_train()
    app.run(port=5000)
