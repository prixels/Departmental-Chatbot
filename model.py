# Import necessary libraries
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

# Load and preprocess JSON data
def load_and_process_json():
    with open('converted_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file) 

    # Initialize lists for training data
    X = []
    y = []
    tags = []
    patterns = []
    responses = []

    # Initialize tokenizer and label encoder
    tokenizer = Tokenizer()
    le = LabelEncoder()

    # Loop through the intents and process data
    for intent in data['intents']:
        for pattern in intent['patterns']:
            X.append(pattern)  # Add the pattern
            y.append(intent['tag'])  # Add the associated tag
            patterns.append(pattern)
            responses.append(intent['responses'])

    # Tokenize the patterns
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    
    # Pad the sequences to ensure equal input length
    max_length = max([len(x) for x in X])  # find the max length of sequences
    X = pad_sequences(X, maxlen=max_length, padding='post')

    # Encode the labels
    y = le.fit_transform(y)
    y = np.array(y)
    
    # One-hot encode the labels
    y = np.eye(len(le.classes_))[y]

    return X, y, tokenizer, le, max_length

# Build the model
def build_model(input_shape, output_shape, tokenizer):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=input_shape))  # Embedding layer
    model.add(LSTM(64, return_sequences=False))  # LSTM layer
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(128, activation='relu'))  # Dense layer
    model.add(Dense(output_shape, activation='softmax'))  # Output layer with softmax for multi-class classification
###512 layers
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model

# Train and save the model
def train_and_save_model():
    # Load and process data
    X, y, tokenizer, label_encoder, max_length = load_and_process_json()

    # Build the model
    model = build_model(X.shape[1], y.shape[1], tokenizer)

    # Train the model
    history = model.fit(X, y, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

    # Save the model
    model.save("chatbot_model.h5")
    print("Model saved to chatbot_model.h5")

    # Save the tokenizer and label encoder for later use
    with open("tokenizer.json", "w") as token_file:
        json.dump(tokenizer.to_json(), token_file)
    
    with open("label_encoder.pkl", "wb") as le_file:
        pickle.dump(label_encoder, le_file)

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot loss curves
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    train_and_save_model()
