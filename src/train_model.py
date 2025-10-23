import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

# === Load dataset ===
df = pd.read_csv("data/twitter_dataset.csv")

# Adjust to your dataset structure
df.columns = ["id", "topic", "sentiment", "text"]
df = df[["text", "sentiment"]].dropna()

# === Preprocess ===
df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["sentiment"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TF-IDF Vectorizer ===
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === Train Models ===
print("Training Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred):.2f}")

print("Training SVM...")
svm = LinearSVC(max_iter=10000)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.2f}")

# === Save the better model (SVM) ===
joblib.dump(svm, "models/svm_sentiment_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("✅ Model and vectorizer saved in /models")

# === Report ===
print("\nClassification Report (SVM):")
print(classification_report(y_test, svm_pred))
