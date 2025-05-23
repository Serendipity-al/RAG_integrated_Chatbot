import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import nltk
import shutil
import os

# Find and clear NLTK data directory
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if os.path.exists(nltk_data_dir):
    shutil.rmtree(nltk_data_dir)

# Redownload everything cleanly
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Tokenize and collect words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist = nltk.word_tokenize(pattern)
        words.extend(wordlist)
        documents.append((wordlist, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))  # 🔄 Fix: previously `set(classes)` was mistakenly used
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]  # 🔄 Fix: was wrongly set as documents[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert to numpy
random.shuffle(training)
training = np.array(training, dtype=np.float32)


# Split into input and output
trainX = np.array(list(training[:, :len(words)]))
trainY = np.array(list(training[:, len(words):]))

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))  # 🔄 Fix: tf.keras.layer → tf.keras.layers
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

# Compile model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot.h5')
print("Executed")
