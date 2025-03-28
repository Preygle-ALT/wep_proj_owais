from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Huggingface DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a chat history to maintain context
chat_history_ids = None


@app.route('/')
def home():
    return "Flask server is running. Use the /chat endpoint to interact with the bot."



@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    data = request.get_json()
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({'response': "Please enter a message."}), 400

    # Encode user input and append to chat history
    input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate a response
    with torch.no_grad():
        if chat_history_ids is not None:
            bot_output = model.generate(
                input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        else:
            bot_output = model.generate(
                input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode bot response
    response = tokenizer.decode(
        bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append the response to chat history
    chat_history_ids = bot_output

    return jsonify({'response': response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
