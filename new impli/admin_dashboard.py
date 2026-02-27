from flask import Flask, render_template
import firebase_admin
from firebase_admin import credentials, firestore
import datetime

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

@app.route("/")
def show_learned_intents():
    docs = db.collection("learned_intents").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    intents = [doc.to_dict() for doc in docs]
    return render_template("dashboard.html", intents=intents)

if __name__ == "__main__":
    app.run(port=5002, debug=True)
