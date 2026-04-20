import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/processed/train.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training split: {DATA_PATH}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    texts = df["combined_text"].fillna("").tolist()
    labels = df["label"].astype(int).tolist()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(ARTIFACTS_DIR / "faiss.index"))
    metadata = [
        {"text": txt, "label": int(lbl)}
        for txt, lbl in zip(texts, labels)
    ]
    with (ARTIFACTS_DIR / "retrieval_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print(f"Saved FAISS index with {len(metadata)} records.")


if __name__ == "__main__":
    main()
