import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_message(message, threshold=0.5):
    message_tfidf = vectorizer.transform([message])
    probability = model.predict_proba(message_tfidf)[0][1]
    prediction = 1 if probability > threshold else 0
    return prediction, probability


if __name__ == "__main__":
    sample_message = input("Enter message: ")
    pred, prob = predict_message(sample_message, threshold=0.3)

    print("\nMessage:", sample_message)
    print("Spam Probability:", round(prob, 4))
    print("Prediction:", "Spam" if pred == 1 else "Not Spam")