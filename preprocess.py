from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

lemmatizer = WordNetLemmatizer()

# Prepare lists
patterns, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])

# Tokenize and lemmatize
words = []
for pattern in patterns:
    tokens = nltk.word_tokenize(pattern)
    words.extend([lemmatizer.lemmatize(token.lower()) for token in tokens])

# Remove duplicates
words = sorted(set(words))
tags = sorted(set(labels))

# Encode data
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert to numerical data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)
X = np.array(X, dtype=object)
y = to_categorical(encoded_labels)

# Pad sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, padding="post")
