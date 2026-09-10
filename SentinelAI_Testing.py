import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

file_path = Path(__file__).resolve().parent / "CEAS_08.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 2. PREPARE THE DATA
# --------------------------------------------------

# Remove rows that do not have a valid label
df = df.dropna(subset=["label"]).copy()

# Fill missing subject/body values with empty text
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

# Combine the subject and body into one text column
df["text"] = df["subject"] + " " + df["body"]

# Remove exact duplicate emails with the same label
df = df.drop_duplicates(subset=["text", "label"])

# X = email text, y = correct phishing label
X = df["text"]
y = df["label"].astype(int)

print("Dataset after cleaning:", df.shape)
print("\nLabel counts:")
print(y.value_counts())

# --------------------------------------------------
# 3. SPLIT INTO TRAINING AND TESTING DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining emails:", len(X_train))
print("Testing emails:", len(X_test))

# --------------------------------------------------
# 4. BUILD THE MACHINE LEARNING MODEL
# --------------------------------------------------

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# --------------------------------------------------
# 5. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining the model...")
model.fit(X_train, y_train)

print("Training complete!")

# --------------------------------------------------
# 6. TEST THE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 7. DISPLAY THE RESULTS
# --------------------------------------------------

print("\nMODEL TEST RESULTS")
print("------------------------------")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
