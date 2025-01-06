from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
import os
import torch
import threading
import json

app = Flask(__name__)

# Load model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

# Context management
context_window = []  # Stores past messages and responses
max_context_tokens = 1500  # Reserve space for context to fit within 2048 tokens

# Path to training.json
training_file_path = "./luma-memory/training/training.json"

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

def start_fine_tuning():
    """Run fine-tuning in a background thread."""
    threading.Thread(target=fine_tune_model).start()

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

    # Custom callback class to handle training progress
    class LogTrainingProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            print("Training started...")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"Training Step {state.global_step} - Loss: {logs.get('loss', 'N/A')}")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Add the custom callback to monitor training progress
    trainer.add_callback(LogTrainingProgressCallback)

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

    # Retrieve relevant training data and integrate them into context
    training_data = load_training_data()  # Load the training data each time (could be optimized)
    memory_integration = " ".join([entry[1] for entry in training_data])  # Use the AI responses as context
    conversation_input = memory_integration + " " + " ".join(context_window)

    # Prepare full conversation input
    inputs = tokenizer(conversation_input, return_tensors="pt")

    # Generate a response
    outputs = model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add the model's response to the context
    context_window.append(response)

    # Fine-tune the model after this interaction
    start_fine_tuning()

    return jsonify({"response": response})

if __name__ == "__main__":
    # Load and process training data before starting the app
    training_data = load_training_data()
    for user_input, ai_response in training_data:
        # Fine-tune with the pre-existing training data
        start_fine_tuning()
    
    app.run(port=5000)
