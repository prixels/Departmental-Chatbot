import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import pad_sequences

# Load the intents JSON file
with open('converted_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize empty lists for patterns, responses, and tags
patterns = []
responses = []
tags = []

# Loop through the intents and collect patterns and tags
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
    tags.append(intent['tag'])

# Tokenizing the patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(X, padding='post')

# Label Encoding the tags
encoder = LabelEncoder()
y = encoder.fit_transform(tags)

# Reshape X to be 3D as required by LSTM (samples, timesteps, features)
X = np.array(X)
X = np.expand_dims(X, axis=-1)  # This adds the "features" dimension

# Create and train the neural network model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

# Save the tokenizer and encoder
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

np.save('label_encoder_classes.npy', encoder.classes_)

print("Model training complete and saved.")
