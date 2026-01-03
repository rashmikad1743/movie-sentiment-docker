# model_training.py

import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib


def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Load dataset from CSV.
    - If path is given via --data, use that.
    - Otherwise, try common default locations.
    - Also handles IMDB column names: review, sentiment -> text, label
    """

    # If user did not pass --data, try these default paths
    if path is None:
        candidate_paths = [
            "data/IMDB Dataset.csv",
            "data/IMDB_Dataset.csv",
            "IMDB Dataset.csv",
            "IMDB_Dataset.csv",
            "data/reviews.csv",
        ]

        for p in candidate_paths:
            if os.path.exists(p):
                path = p
                print(f"[INFO] Using dataset: {path}")
                break

        if path is None:
            raise FileNotFoundError(
                f"Could not find dataset. Tried: {candidate_paths}. "
                f"Provide path with --data."
            )
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        print(f"[INFO] Using dataset from --data: {path}")

    # Read CSV
    df = pd.read_csv(path)

    # Rename IMDB style columns to our standard names
    rename_map = {}
    if "review" in df.columns:
        rename_map["review"] = "text"
    if "sentiment" in df.columns:
        rename_map["sentiment"] = "label"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Now we expect 'text' and 'label'
    required = ["text", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset must have columns {required}. "
            f"Missing columns: {missing}. Actual columns: {list(df.columns)}"
        )

    # Drop rows with empty text or label
    df = df.dropna(subset=["text", "label"])

    print(f"[INFO] Loaded {len(df)} rows from dataset.")
    print(df.head(3))

    return df


def train_model(data_path: str | None = None) -> None:
    # 1) Load data
    df = load_data(data_path)

    X = df["text"]
    y = df["label"]

    # 2) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3) Build pipeline: TF-IDF + Logistic Regression
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,  # IMDB is big, so use more features
                    ngram_range=(1, 2),  # unigrams + bigrams
                    stop_words="english",
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # 4) Evaluate on TRAIN data
    print("[INFO] Evaluating on training set...")
    y_pred_train = model.predict(X_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    train_prec = precision_score(y_train, y_pred_train, pos_label="positive")
    train_rec = recall_score(y_train, y_pred_train, pos_label="positive")
    train_f1 = f1_score(y_train, y_pred_train, pos_label="positive")

    print("\n----- TRAIN SET Evaluation -----")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Train Precision: {train_prec:.4f}")
    print(f"Train Recall   : {train_rec:.4f}")
    print(f"Train F1 Score : {train_f1:.4f}")

    # 5) Evaluate on TEST data
    print("\n----- TEST SET Evaluation -----")
    y_pred_test = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, pos_label="positive")
    test_rec = recall_score(y_test, y_pred_test, pos_label="positive")
    test_f1 = f1_score(y_test, y_pred_test, pos_label="positive")

    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test Precision : {test_prec:.4f}")
    print(f"Test Recall    : {test_rec:.4f}")
    print(f"Test F1 Score  : {test_f1:.4f}")

    # 6) Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/sentiment_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset CSV. Example: --data 'data/IMDB Dataset.csv'",
    )
    args = parser.parse_args()

    train_model(args.data)
