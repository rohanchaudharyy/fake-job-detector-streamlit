import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DATA_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train_df["combined_text"].fillna(""))
    y_train = train_df["label"].astype(int)

    model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

    x_test = vectorizer.transform(test_df["combined_text"].fillna(""))
    y_test = test_df["label"].astype(int)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
