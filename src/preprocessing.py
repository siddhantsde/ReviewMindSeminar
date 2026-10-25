"""Review cleaning, validation, and duplicate detection."""
import pandas as pd

def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return accepted and rejected rows; no invalid input is silently discarded."""
    data = df.copy()
    defaults={"product_id":"","category":"Uncategorized","review_date":pd.NaT,"verified_purchase":False,"helpful_votes":0}
    for column,value in defaults.items():
        if column not in data: data[column]=value
    raw_rating = data["rating"].copy()
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    rating_missing = raw_rating.isna() | raw_rating.astype(str).str.strip().eq("")
    data["review_text"] = data["review_text"].fillna("").astype(str).str.strip()
    data["product_name"] = data["product_name"].fillna("").astype(str).str.strip()
    # Blank ratings are legitimate for copied review text with no star score.
    # Nonblank values outside the 1–5 scale are rejected rather than guessed.
    good = (data.rating.between(1, 5) | rating_missing) & data.review_text.ne("") & data.product_name.ne("")
    accepted, rejected = data.loc[good].copy(), data.loc[~good].copy()
    accepted["rating"] = accepted["rating"].astype("Int64")
    accepted["review_text_clean"] = accepted.review_text.str.replace(r"\s+", " ", regex=True)
    accepted["review_date"] = pd.to_datetime(accepted["review_date"], errors="coerce")
    accepted["helpful_votes"] = pd.to_numeric(accepted["helpful_votes"], errors="coerce").fillna(0).astype(int)
    accepted["verified_purchase"] = accepted["verified_purchase"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    accepted["is_duplicate"] = accepted.review_text_clean.str.lower().duplicated(keep="first")
    return accepted, rejected
