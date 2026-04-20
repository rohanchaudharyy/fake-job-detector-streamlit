import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from src.features.extract_features import extract_rule_signals, signals_to_list


ARTIFACTS_DIR = Path("artifacts")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class HybridDetector:
    def __init__(self) -> None:
        self.model = joblib.load(ARTIFACTS_DIR / "model.joblib")
        self.vectorizer = joblib.load(ARTIFACTS_DIR / "vectorizer.joblib")
        self.index = faiss.read_index(str(ARTIFACTS_DIR / "faiss.index"))
        with (ARTIFACTS_DIR / "retrieval_metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.ollama_client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def _retrieve_similar(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        query = self.embed_model.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            hits.append(
                {
                    "similarity": float(score),
                    "label": int(item["label"]),
                    "text": item["text"][:450],
                }
            )
        return hits

    def _explain_with_ollama(
        self,
        input_text: str,
        predicted_label: int,
        confidence: float,
        rule_flags: list[str],
        similar_examples: list[dict[str, Any]],
    ) -> str:
        label_txt = "FAKE" if predicted_label == 1 else "REAL"
        prompt = (
            "You are a fraud risk analyst for job postings.\n"
            "Write a formal, concise explanation.\n"
            f"Predicted class: {label_txt}\n"
            f"Confidence: {confidence:.4f}\n"
            f"Rule flags: {rule_flags}\n"
            f"Similar examples: {similar_examples}\n"
            f"Input job listing: {input_text}\n\n"
            "Return: 1 short verdict paragraph and 3 bullet reasons grounded in the evidence."
        )
        try:
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as exc:
            return (
                "Explanation service unavailable. "
                f"Model prediction and retrieval evidence are still valid. ({exc})"
            )

    def predict(self, job_text: str, company_profile: str = "", salary_range: str = "", contact_info: str = "") -> dict[str, Any]:
        vec = self.vectorizer.transform([job_text])
        proba_fake = float(self.model.predict_proba(vec)[0][1])
        label = 1 if proba_fake >= 0.5 else 0

        signals = extract_rule_signals(job_text, company_profile, salary_range, contact_info)
        rule_list = signals_to_list(signals)
        similar = self._retrieve_similar(job_text, top_k=int(os.getenv("TOP_K_RETRIEVAL", "5")))
        explanation = self._explain_with_ollama(job_text, label, proba_fake, rule_list, similar)

        return {
            "prediction": "fake" if label == 1 else "real",
            "confidence_fake": proba_fake,
            "rule_signals": asdict(signals),
            "rule_findings": rule_list,
            "similar_examples": similar,
            "explanation": explanation,
        }
