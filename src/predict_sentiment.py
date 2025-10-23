import joblib
from preprocess import clean_text

# === Load model and vectorizer ===
model = joblib.load("models/svm_sentiment_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

def predict_sentiment(text: str) -> str:
    """
    Predicts sentiment (Positive, Negative, Neutral) for given input text.
    """
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction
