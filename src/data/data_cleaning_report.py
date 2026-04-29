from pathlib import Path

import pandas as pd


RAW_DATASET = Path("data/raw/fake_job_dataset.csv")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def main() -> None:
    if not RAW_DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {RAW_DATASET}")

    df = pd.read_csv(RAW_DATASET)
    print("=== DATA CLEANING REPORT ===")
    print(f"Rows (raw): {len(df)}")
    print(f"Columns: {list(df.columns)}")

    text_cols = ["title", "description", "requirements", "company_profile", "salary_range"]
    for col in text_cols:
        if col in df.columns:
            missing_before = int(df[col].isna().sum())
            df[col] = df[col].apply(clean_text)
            missing_after = int((df[col].str.strip() == "").sum())
            print(f"{col}: missing_before={missing_before}, empty_after_cleaning={missing_after}")

    if "fraudulent" in df.columns:
        raw_missing = int(df["fraudulent"].isna().sum())
        normalized = (
            df["fraudulent"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"t": "1", "f": "0", "true": "1", "false": "0"})
        )
        label = pd.to_numeric(normalized, errors="coerce")
        valid_rows = int(label.notna().sum())
        print(f"fraudulent: missing_raw={raw_missing}, valid_after_normalization={valid_rows}")

    print("Cleaning logic mirrors src/data/preprocess.py")


if __name__ == "__main__":
    main()
