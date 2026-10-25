"""CSV loading and schema normalization."""
from pathlib import Path
import pandas as pd

REQUIRED = {"product_name", "rating", "review_text"}
OPTIONAL = ["product_id", "category", "review_date", "verified_purchase", "helpful_votes"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out

def load_reviews(source) -> pd.DataFrame:
    """Load a CSV path or file-like object and validate required fields."""
    df = normalize_columns(pd.read_csv(source))
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    for col, default in [("product_id", ""), ("category", "Uncategorized"), ("review_date", pd.NaT),
                         ("verified_purchase", False), ("helpful_votes", 0)]:
        if col not in df: df[col] = default
    if "review_id" not in df: df["review_id"] = [f"upload-{i+1}" for i in range(len(df))]
    return df

def load_sample(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "sample_reviews.csv"
    return load_reviews(path)
