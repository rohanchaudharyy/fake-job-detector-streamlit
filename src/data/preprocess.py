from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH = Path("data/raw/fake_job_postings.csv")
OUT_DIR = Path("data/processed")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["description", "company_profile", "salary_range", "requirements", "fraudulent"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    for col in ["description", "company_profile", "salary_range", "requirements", "title", "location"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    df["combined_text"] = (
        df.get("title", "")
        + " "
        + df.get("description", "")
        + " "
        + df.get("requirements", "")
    ).str.strip()

    # Some dataset exports include missing labels and/or boolean strings ("t"/"f").
    # Keep only rows with valid labels and normalize to integer 0/1.
    df = df[df["fraudulent"].notna()].copy()
    normalized = (
        df["fraudulent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"t": "1", "f": "0", "true": "1", "false": "0"})
    )
    df["label"] = pd.to_numeric(normalized, errors="coerce")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    return df


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. Place your dataset CSV there first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_PATH)
    df = preprocess_dataframe(raw_df)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"Saved processed splits to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
