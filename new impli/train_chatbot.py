import nltk
import json, pickle, numpy as np, random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# Download resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load intents
with open('intents.json', encoding='utf-8') as f:
    intents = json.load(f)

# Prepare data
corpus, labels, classes = [], [], []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern.lower())
        filtered = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
        corpus.append(" ".join(filtered))
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

classes = sorted(list(set(classes)))

# Save classes
pickle.dump(classes, open('classes.pkl', 'wb'))

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

# Convert labels to one-hot
y = np.zeros((len(labels), len(classes)))
for i, label in enumerate(labels):
    y[i][classes.index(label)] = 1

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=SGD(learning_rate=0.01, momentum=0.9), 
              metrics=['accuracy'])

# Train model with early stopping
model.fit(X, y, epochs=300, batch_size=16, verbose=1,
          callbacks=[EarlyStopping(monitor='loss', patience=10)])

model.save('chatbot_model.h5')
print("✅ Model trained and saved successfully!")
