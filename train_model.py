"""Train the spam classifier and save it to disk.

Run once (or whenever you want to retrain):
    python train_model.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data") / "SMSSpamCollection"
MODEL_PATH = Path("model") / "spam_model.joblib"
RANDOM_SEED = 42


def load_data(path):
    """Load the tab-separated file: <label>\t<message>, no header row."""
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at '{path}'.\n"
            "Please download it first (see README.md)."
        )
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        quoting=3,  # SMS text contains quote characters; don't parse them
    )
    df = df.dropna()
    df = df[df["label"].isin(["spam", "ham"])]
    return df


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages.")
    print(df["label"].value_counts().to_string(), "\n")

    # Stratified 80/20 split with a fixed seed.
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("nb", MultinomialNB()),
        ]
    )

    # Fit ONLY on the training data.
    pipeline.fit(X_train, y_train)

    # Evaluate on the held-out test data.
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="spam")
    recall = recall_score(y_test, predictions, pos_label="spam")

    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Spam precision:   {precision:.4f}")
    print(f"Spam recall:      {recall:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
