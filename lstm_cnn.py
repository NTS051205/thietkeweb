# Importing essential libraries and functions

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense
from sklearn.model_selection import train_test_split

# Load training dataset from Project8_SentimentAnalysis_with_NeuralNetwork/a1_IMDB_Dataset.csv
movie_reviews = pd.read_csv("Project8_SentimentAnalysis_with_NeuralNetwork/a1_IMDB_Dataset.csv")

# Preprocess text function
def preprocess_text(sen):
    '''Cleans text data: removes HTML tags, punctuations, numbers, and stopwords.'''
    TAG_RE = re.compile(r'<[^>]+>')
    sentence = TAG_RE.sub('', sen)  # Remove HTML tags
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)  # Remove punctuations and numbers
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # Remove single characters
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove multiple spaces
    sentence = sentence.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    sentence = ' '.join(word for word in sentence.split() if word not in stop_words)  # Remove stopwords
    return sentence

# Preprocess the text column
movie_reviews['review'] = movie_reviews['review'].apply(preprocess_text)

# Convert sentiment labels to binary (0 and 1)
movie_reviews['sentiment'] = movie_reviews['sentiment'].apply(lambda x: 1 if x == "positive" else 0)

# Prepare data for the model
X = movie_reviews['review']
y = movie_reviews['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
vocab_size = 5000
max_len = 100

word_tokenizer = Tokenizer(num_words=vocab_size)
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)

# Pad sequences to the same length
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# Build the LSTM + CNN model
# Build the LSTM + CNN model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(LSTM(64, return_sequences=True))  # Đảm bảo LSTM nhận đúng định dạng
model.add(GlobalMaxPooling1D())  # Tùy chọn nếu muốn giảm chiều
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict sentiments for new data
# new_reviews = pd.read_excel("nguoihoc.xlsx")
# new_reviews['text'] = new_reviews['text'].apply(preprocess_text)
# Load new data
new_reviews = pd.read_excel("nguoihoc.xlsx")

# Clean column names
new_reviews.columns = new_reviews.columns.str.strip()

# Preprocess text
new_reviews['text'] = new_reviews['text'].apply(preprocess_text)

# Proceed with predictions...

new_sequences = word_tokenizer.texts_to_sequences(new_reviews['text'])
new_padded = pad_sequences(new_sequences, padding='post', maxlen=max_len)
new_predictions = model.predict(new_padded)

# Save predictions back to the file
new_reviews['sentiment'] = (new_predictions > 0.5).astype(int)
new_reviews.to_excel("nguoihoc_predictions.xlsx", index=False, columns=['stt', 'text', 'sentiment'])
print("Predictions saved to nguoihoc_predictions.xlsx")
