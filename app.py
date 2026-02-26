from flask import Flask, request, render_template
import pickle
import os

# Load model and vectorizer once at startup
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

THRESHOLD = 0.3  # you can adjust this

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    message_tfidf = vectorizer.transform([message])
    probability = model.predict_proba(message_tfidf)[0][1]

    prediction = 1 if probability > THRESHOLD else 0

    print("Message:", message)
    print("Spam Probability:", probability)
    print("Threshold:", THRESHOLD)
    print("Final Prediction:", "Spam" if prediction == 1 else "Not Spam")
    print("-" * 40)

    return render_template(
        "index.html",
        prediction="Spam" if prediction == 1 else "Not Spam",
        probability=round(float(probability), 4),
        message=message
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)