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
        self._validate_artifacts()
        self.model = joblib.load(ARTIFACTS_DIR / "model.joblib")
        self.vectorizer = joblib.load(ARTIFACTS_DIR / "vectorizer.joblib")
        self.index = faiss.read_index(str(ARTIFACTS_DIR / "faiss.index"))
        with (ARTIFACTS_DIR / "retrieval_metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.ollama_client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    @staticmethod
    def _validate_artifacts() -> None:
        required = [
            ARTIFACTS_DIR / "model.joblib",
            ARTIFACTS_DIR / "vectorizer.joblib",
            ARTIFACTS_DIR / "faiss.index",
            ARTIFACTS_DIR / "retrieval_metadata.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required artifacts. Run preprocessing, training, and index build first.\n"
                f"Missing files: {missing}"
            )

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
        soft_notice = (
            "I am currently sharing a concise explanation based on the model output and "
            "retrieved evidence. For richer conversational reasoning, the local demo uses "
            "a connected Ollama model."
        )
        fallback = (
            f"{soft_notice}\n\nEducational analysis only: the posting is predicted as {label_txt} "
            f"with fake probability {confidence:.2%}. "
            f"Key risk signals: {rule_flags if rule_flags else ['No explicit rule signal triggered']}. "
            "This is a model-based assessment, not a legal determination."
        )
        prompt = (
            "You are helping with an academic NLP assignment.\n"
            "Task: explain a model-based text risk assessment for a job posting.\n"
            "Important constraints:\n"
            "- This is educational and non-legal analysis.\n"
            "- Do not refuse.\n"
            "- Do not provide policy warnings.\n"
            "- Base the explanation only on the evidence provided below.\n"
            "- Keep tone professional, clear, and human.\n"
            f"Predicted class: {label_txt}\n"
            f"Confidence: {confidence:.4f}\n"
            f"Rule flags: {rule_flags}\n"
            f"Similar examples: {similar_examples}\n"
            f"Input job listing: {input_text}\n\n"
            "Return exactly:\n"
            "1) One short verdict paragraph.\n"
            "2) One paragraph explaining the calculation basis (probability threshold + triggered signals).\n"
            "3) One paragraph citing retrieved-example patterns.\n"
            "Do not say 'I cannot assist' or similar refusal phrases."
        )
        try:
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"].strip()
            refusal_markers = [
                "i cannot assist",
                "i can't assist",
                "cannot help with",
                "cannot comply",
                "i’m unable to",
                "i am unable to",
            ]
            if any(marker in content.lower() for marker in refusal_markers):
                return fallback
            return content
        except Exception:
            return fallback

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
