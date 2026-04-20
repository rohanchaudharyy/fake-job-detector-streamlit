# Fake Job Detector Streamlit App

A formal, submission-ready Streamlit application that detects potentially fake job postings using a hybrid pipeline:

- ML classification (TF-IDF + Logistic Regression)
- Rule-based suspicious signal checks
- Semantic retrieval of similar known postings (FAISS)
- Local LLM explanation and chat via Ollama

## 1) Project Objective

Given a job listing, the application:

1. predicts whether it is likely `fake` or `real`
2. provides confidence and triggered risk signals
3. retrieves similar examples from known data
4. generates an evidence-grounded explanation in a chat interface

## 2) Directory Structure

```text
fake-job-detector-streamlit/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── extract_features.py
│   ├── inference/
│   │   └── predict_and_explain.py
│   ├── models/
│   │   └── train_baseline.py
│   └── retrieval/
│       └── build_faiss_index.py
├── tests/
│   ├── test_features.py
│   └── test_inference.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 3) Setup

### Prerequisites

- Python 3.10+
- Ollama installed and running locally
- A local Ollama model pulled (for example: `ollama pull llama3.1:8b`)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and edit if needed:

```bash
cp .env.example .env
```

## 4) Data Preparation

Place your raw dataset CSV (for example, Kaggle fake job postings) under:

- `data/raw/fake_job_postings.csv`

Then run:

```bash
python -m src.data.preprocess
```

This creates cleaned split files under `data/processed`.

## 5) Model Training

```bash
python -m src.models.train_baseline
```

This produces:

- `artifacts/model.joblib`
- `artifacts/vectorizer.joblib`
- `artifacts/metrics.json`

## 6) Build Retrieval Index

```bash
python -m src.retrieval.build_faiss_index
```

This produces:

- `artifacts/faiss.index`
- `artifacts/retrieval_metadata.json`

## 7) Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## 8) Evaluation Metrics

The baseline training script reports:

- Accuracy
- Precision
- Recall
- F1-score

For fraud detection, prioritize recall/F1 for the fake class.

## 9) Notes for Submission

- Keep all generated artifacts reproducible via scripts.
- Include screenshot(s) of the Streamlit app in your report.
- Explain why hybrid design improves trust:
  - ML for deterministic decision
  - rules for explicit risk flags
  - retrieval for evidence
  - LLM for user-friendly explanation
