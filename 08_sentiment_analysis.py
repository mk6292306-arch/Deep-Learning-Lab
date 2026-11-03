import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class TextReviewClassifier:
    def __init__(self, vocab_size=10000, max_length=200, embedding_dim=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None
        
    def preprocess_text(self, text):
        """Preprocess text by removing special characters, stopwords, and lemmatizing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
       
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def load_and_preprocess_data(self, file_path=None):
        """Load and preprocess data. If no file is provided, use IMDB dataset"""
        if file_path:
            # Load your custom dataset
            # Expected format: CSV with 'text' and 'sentiment' columns
            # Sentiment: 0 for negative, 1 for positive
            df = pd.read_csv(file_path)
            texts = df['text'].values
            labels = df['sentiment'].values
        else:
            # Use IMDB dataset as example
            print("Loading IMDB dataset...")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=self.vocab_size)
            
            # Get word index to reverse the sequences
            word_index = tf.keras.datasets.imdb.get_word_index()
            reverse_word_index = {value: key for (key, value) in word_index.items()}
            
            # Convert sequences back to text
            def decode_review(encoded_review):
                return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])
            
            # Combine train and test for full dataset
            x = np.concatenate([x_train, x_test])
            y = np.concatenate([y_train, y_test])
            
            # Decode a subset for faster processing
            texts = [decode_review(review) for review in x[:5000]]
            labels = y[:5000]
        
        # Preprocess texts
        print("Preprocessing texts...")
        texts = [self.preprocess_text(text) for text in texts]
        
        return texts, labels
    
    def prepare_data(self, texts, labels):
        """Tokenize and pad sequences"""
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
     
   # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, labels, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """Build the neural network model"""
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model"""
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_classes))
        
        
# Confusion matrix
        cm = confusion_matrix(y_test, predicted_classes)
        plt.figure(figsize=(6, 4))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        return loss, accuracy, predictions
    
    def predict_text(self, text):
        """Predict sentiment for a new text"""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
        
        # Predict
        prediction = self.model.predict(padded)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        confidence = prediction[0][0] if sentiment == "Positive" else 1 - prediction[0][0]
        
        return sentiment, confidence
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
if _name_ == "_main_":
    # Initialize classifier
    classifier = TextReviewClassifier(
        vocab_size=10000,
        max_length=200,
        embedding_dim=100
    )
    
    # Load and preprocess data
    # If you have your own dataset, provide the file path:
    # texts, labels = classifier.load_and_preprocess_data("your_dataset.csv")
    texts, labels = classifier.load_and_preprocess_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(texts, labels)
    
    # Build model
    model = classifier.build_model()
    model.summary()
    
    # Train model
    print("Training model...")
    history = classifier.train(X_train, y_train, X_test, y_test, epochs=5)
    
    # Evaluate model
    print("Evaluating model...")
    loss, accuracy, predictions = classifier.evaluate(X_test, y_test)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save model
    model.save("sentiment_analysis_model.h5")
    print("Model saved as 'sentiment_analysis_model.h5'")
    
    # Example predictions
    sample_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The acting was terrible and the plot was boring. I hated it.",
        "It was okay, not great but not terrible either.",
        "This product is amazing and works perfectly. Highly recommend!",
        "Poor quality and waste of money. Would not buy again."
    ]
    print("\nSample Predictions:")
    for review in sample_reviews:
        sentiment, confidence = classifier.predict_text(review)
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
        print()
