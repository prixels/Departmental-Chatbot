import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download NLTK resources if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Load your JSON file
# Use UTF-8 encoding to load the JSON data
with open('converted_data.json', 'r', encoding='utf-8') as file:
    intents = json.load(file) 

# Function to preprocess and tokenize sentences
def preprocess_sentence(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    # Remove punctuation
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    # Tokenize the sentence
    words = word_tokenize(sentence)
    # Remove stopwords (optional)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# Build vocabulary (list of unique words)
all_words = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:  # Assuming `patterns` contains user inputs
        words = preprocess_sentence(pattern)
        all_words.extend(words)

# Get unique words and sort them
vocabulary = sorted(set(all_words))

# Save the vocabulary as a NumPy array
np.save("words.npy", vocabulary)
print(f"Vocabulary saved as 'words.npy' with {len(vocabulary)} unique words!")

# Check saved vocabulary
loaded_words = np.load("words.npy", allow_pickle=True)
print("Loaded Vocabulary:", loaded_words)
