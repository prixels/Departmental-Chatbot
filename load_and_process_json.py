import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load JSON data
with open('converted_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file) 

# Extract patterns and tags
patterns = []
tags = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Tokenize patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)

# Pad sequences to the same length
X = pad_sequences(X, padding="post")  # Pads sequences with zeros at the end

# Save tokenizer
import pickle
with open("tokenizer.json", "wb") as f:
    pickle.dump(tokenizer, f)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)
y = to_categorical(y)  # Convert to one-hot encoding

# Save label encoder classes
np.save("label_encoder_classes.npy", label_encoder.classes_)

# Print shapes to confirm
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
