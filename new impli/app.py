from flask import Flask, request, jsonify
import random, json, pickle, numpy as np, re, requests, datetime, threading, time
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
from multi_ai_response import generate_best_response
from deep_translator import GoogleTranslator

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
model = load_model('chatbot_model.h5')
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = vectorizer.get_feature_names_out()
classes = pickle.load(open('classes.pkl', 'rb'))

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ================= NLP & Prediction =================
def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
    return ' '.join(tokens)

def predict_class(sentence):
    cleaned = clean_text(sentence)
    vector = vectorizer.transform([cleaned]).toarray()
    preds = model.predict(vector, verbose=0)[0]
    results = [{"intent": classes[i], "probability": str(p)} for i, p in enumerate(preds) if p > 0.25]
    results.sort(key=lambda x: float(x["probability"]), reverse=True)
    return results

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure how to respond to that."

# ================= Utility Features =================
def get_weather(city):
    try:
        api_key = "c290c0b98b64a4c576c0164601338cab"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        return f"The weather in {city.title()} is {res['weather'][0]['description']} with {res['main']['temp']}°C."
    except:
        return "Couldn't fetch weather."

def get_joke():
    try:
        res = requests.get("https://official-joke-api.appspot.com/jokes/random").json()
        return f"{res['setup']} ... {res['punchline']}"
    except:
        return "Couldn't fetch a joke."

def get_quote():
    try:
        res = requests.get("https://api.quotable.io/random").json()
        return f"\"{res['content']}\" — {res['author']}"
    except:
        return "Couldn't fetch a quote."

def get_time():
    return datetime.datetime.now().strftime("It's %I:%M %p.")

def get_date():
    return datetime.datetime.today().strftime("Today is %B %d, %Y.")

def solve_math_expression(expression):
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"The answer is {result}"
    except:
        return None

def convert_currency(message):
    try:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(\w+)\s+to\s+(\w+)", message.lower())
        if match:
            amount, from_curr, to_curr = match.groups()
            url = f"https://api.exchangerate.host/convert?from={from_curr.upper()}&to={to_curr.upper()}&amount={amount}"
            res = requests.get(url).json()
            result = res['result']
            return f"{amount} {from_curr.upper()} is {result:.2f} {to_curr.upper()}"
        return None
    except:
        return "Currency conversion failed."

def get_stock_price(msg):
    try:
        match = re.search(r"(?:stock|price)\s+(?:of\s+)?([A-Z]{1,5})", msg.upper())
        if match:
            symbol = match.group(1)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            latest_price = data["Close"].iloc[-1]
            return f"The current stock price of {symbol} is ${latest_price:.2f}."
        return None
    except:
        return "Couldn't fetch stock data."

def translate_text(message, target_lang="en"):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(message)
    except Exception as e:
        print("Translation error:", e)
        return "Sorry, I couldn't translate that."

def log_chat(user_id, user_message, bot_response):
    try:
        db.collection("chat_logs").add({
            "user_id": user_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": datetime.datetime.now()
        })
    except:
        pass

def log_ai_learned_intent(tag, pattern, response, source="openai"):
    try:
        db.collection("learned_intents").add({
            "tag": tag,
            "pattern": pattern,
            "response": response,
            "source": source,
            "timestamp": datetime.datetime.now(),
            "approved": False,
            "reviewed": False
        })
    except Exception as e:
        print("Error logging learned intent:", e)

def reminder_worker():
    while True:
        try:
            now = datetime.datetime.now()
            reminders = db.collection("reminders") \
                .where("time", "<=", now) \
                .where("notified", "==", False) \
                .stream()

            for reminder in reminders:
                data = reminder.to_dict()
                log_chat(data["user_id"], "[Reminder Triggered]", data["message"])
                db.collection("reminders").document(reminder.id).update({"notified": True})
        except Exception as e:
            print("Reminder check failed:", e)
        time.sleep(60)

threading.Thread(target=reminder_worker, daemon=True).start()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id", "default")
    msg = data.get("message")

    if not msg:
        return jsonify({"response": "Please enter a message."})

    ints = predict_class(msg)

    if not ints:
        ai_data = generate_best_response(msg)
        if ai_data:
            new_intent = {
                "tag": ai_data["tag"],
                "patterns": [msg],
                "responses": [ai_data["response"]],
                "context": [""]
            }
            with open("intents.json", "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["intents"].append(new_intent)
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            log_ai_learned_intent(ai_data["tag"], msg, ai_data["response"], ai_data.get("source", "openai"))
            response = ai_data["response"]
        else:
            response = "Sorry, I couldn’t generate a good response this time."
    else:
        tag = ints[0]['intent']
        if tag == "weather":
            city = msg.split("in")[-1].strip()
            response = get_weather(city)
        elif tag == "joke":
            response = get_joke()
        elif tag == "quote":
            response = get_quote()
        elif tag == "math":
            result = solve_math_expression(msg)
            response = result if result else get_response(ints, intents)
        elif tag == "time":
            response = get_time()
        elif tag == "date":
            response = get_date()
        elif tag == "translate":
            translation = translate_text(msg)
            base_response = get_response(ints, intents)
            response = base_response + " " + translation
        elif "stock" in msg.lower():
            result = get_stock_price(msg)
            response = result if result else get_response(ints, intents)
        elif "to" in msg.lower():
            result = convert_currency(msg)
            response = result if result else get_response(ints, intents)
        else:
            response = get_response(ints, intents)

    log_chat(user_id, msg, response)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)