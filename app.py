import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import logging

app = Flask(__name__)

# Enable logging for Flask
logging.basicConfig(level=logging.DEBUG)

# Load the model and necessary files
try:
    model = load_model('chatbot_model.h5')
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Open 'tokenizer.json' with error handling for encoding issues
try:
    with open('tokenizer.json', 'r', encoding='utf-8', errors='ignore') as f:
        tokenizer = tokenizer_from_json(json.load(f))
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")

# Load the label encoder classes
try:
    label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
except Exception as e:
    logging.error(f"Error loading label encoder classes: {e}")

max_length = 15

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML page

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get user message from POST request
        message = request.json['message']
        logging.debug(f"User message received: {message}")

        # Preprocess message
        sequence = tokenizer.texts_to_sequences([message])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Predict intent
        prediction = model.predict(padded_sequence)
        intent_index = np.argmax(prediction)
        intent = label_encoder_classes[intent_index]

        logging.debug(f"Predicted intent: {intent}")

        # Load responses based on intent
        with open('converted_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        responses = [intent_data['responses'] for intent_data in data['intents'] if intent_data['tag'] == intent]

        if responses:
            response = np.random.choice(responses[0])  # Random response
        else:
            response = "Sorry, I didn't understand that."

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error in /chat route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
