import json, pickle, nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.feature_extraction.text import TfidfVectorizer
from firebase_admin import credentials, firestore, initialize_app
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Firebase setup
cred = credentials.Certificate("firebase_key.json")
initialize_app(cred)
db = firestore.client()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load base intents
with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)

# Load approved learned intents
approved = db.collection("learned_intents")\
    .where("approved", "==", True)\
    .where("reviewed", "==", True)\
    .stream()

added = 0
for doc in approved:
    data = doc.to_dict()
    new_entry = {
        "tag": data["tag"],
        "patterns": [data["pattern"]],
        "responses": [data["response"]],
        "context": [""]
    }
    intents["intents"].append(new_entry)
    added += 1

if added == 0:
    print("No approved intents found. Nothing to retrain.")
    exit()

# Save updated intents.json
with open("intents.json", "w", encoding="utf-8") as f:
    json.dump(intents, f, indent=4)

# Prepare training data
corpus, labels, classes = [], [], []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
        corpus.append(" ".join(filtered))
        labels.append(intent["tag"])
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

y = np.zeros((len(labels), len(classes)))
for i, label in enumerate(labels):
    y[i][classes.index(label)] = 1

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

model.fit(X, y, epochs=300, batch_size=16, verbose=1)

model.save("chatbot_model.h5")
pickle.dump(classes, open("classes.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print(f"✅ Retrained with {added} approved learned intents.")
