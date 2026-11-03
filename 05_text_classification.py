import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load a sample dataset (IMDB reviews for sentiment analysis)
# You can replace this with your own dataset
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Get the word index to reverse the sequences
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# Convert the sequences back to text for demonstration
texts = [decode_review(review) for review in x_train[:2000]]  
# Using first 2000 reviews for speed
labels = y_train[:2000]

# Parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 100
OOV_TOKEN = '<OOV>'

# Preprocessing - create a tokenizer and fit on our texts
tokenizer = Tokenizer(num_words=VOCAB_SIZE, 
oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(texts)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, 
maxlen=MAX_LENGTH, padding='post', truncating='post')

# Convert to numpy array
X = np.array(padded_sequences)
y = np.array(labels)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, 
test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Embedding(VOCAB_SIZE, 
EMBEDDING_DIM, input_length=MAX_LENGTH),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])
# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=5,  # Reduced for demonstration
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# Make prediction function
def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, 
padding='post', truncating='post')
    prediction = model.predict(padded)
    return prediction[0][0]  # Return probability for class 1

# Example prediction
sample_text = "This movie was fantastic and I really enjoyed it!"
print(f"Prediction score for positive text: {predict_text(sample_text):.4f}")

sample_text = "This movie was terrible and I hated every minute of it."
print(f"Prediction score for negative text: {predict_text(sample_text):.4f}")
