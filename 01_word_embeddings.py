import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
# 1. Sample corpus
texts = [
    "Deep learning models learn language",
    "Word embeddings map words to vectors",
    "Neural networks use embeddings for NLP"
]

# 2. Tokenize and convert to integer sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# 3. Pad sequences
data = pad_sequences(sequences, padding='post')

# 4. Labels (dummy, just to train something)
labels = np.array([0, 1, 0])

# 5. Model
model = Sequential([
Embedding(input_dim=len(word_index)+1,output_dim=8,input_length=data.shape[1]),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(data, labels, epochs=10, verbose=0)

# 6. Get embeddings
embeddings = model.layers[0].get_weights()[0]
print("Vocabulary:", word_index)
print("Embedding for 'embeddings':", embeddings[word_index['embeddings']])

