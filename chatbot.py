import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model("chatbot_model.h5")

# Load the tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as tokenizer_file:
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    tokenizer = tokenizer_from_json(json.load(tokenizer_file))

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_encoder_classes.npy", allow_pickle=True)

# Load intents for generating responses
with open("converted_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Function to get a response from the chatbot
def get_response(user_input):
    # Preprocess input
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=model.input_shape[1], padding="post")

    # Predict the intent
    predictions = model.predict(padded_sequence, verbose=0)
    predicted_label_index = np.argmax(predictions)
    predicted_tag = label_encoder.inverse_transform([predicted_label_index])[0]

    # Get the corresponding response
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return np.random.choice(intent["responses"])  # Randomly pick a response

    return "I'm sorry, I didn't understand that."

# Chat loop
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    response = get_response(user_input)
    print(f"Chatbot: {response}")
