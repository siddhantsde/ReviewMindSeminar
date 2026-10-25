"""Dataset-level and product-level metrics."""
import pandas as pd

def summary_metrics(df):
    n = len(df)
    counts = df.sentiment.value_counts() if n and "sentiment" in df else pd.Series(dtype=int)
    avg=df.rating.mean() if n and "rating" in df else float("nan")
    return {"review_count": n, "average_rating": round(float(avg),2) if pd.notna(avg) else None,
            "positive_pct": float(round(100*counts.get("positive",0)/n,1)) if n else 0,
            "neutral_pct": float(round(100*counts.get("neutral",0)/n,1)) if n else 0,
            "negative_pct": float(round(100*counts.get("negative",0)/n,1)) if n else 0,
            "product_count": int(df.product_name.nunique()) if n else 0}

def product_metrics(df):
    if df.empty: return pd.DataFrame(columns=["product_name","review_count","average_rating","negative_pct"])
    return df.groupby("product_name").agg(review_count=("review_text","size"), average_rating=("rating","mean"), negative_pct=("sentiment",lambda s:100*(s=="negative").mean())).reset_index().assign(average_rating=lambda x:x.average_rating.round(2), negative_pct=lambda x:x.negative_pct.round(1))

def key_findings(df, minimum_reviews=10):
    findings=[]
    pm=product_metrics(df)
    eligible=pm[pm.review_count>=minimum_reviews]
    if len(eligible):
        worst=eligible.sort_values("negative_pct",ascending=False).iloc[0]
        findings.append(f"{worst.product_name} has the highest negative sentiment share ({worst.negative_pct:.1f}%) among products with at least {minimum_reviews} reviews (n={worst.review_count}).")
    if len(df):
        if df.rating.notna().any(): findings.append(f"The overall average rating is {df.rating.mean():.2f}/5 across {int(df.rating.notna().sum())} rated reviews.")
        unrated=int(df.rating.isna().sum())
        if unrated: findings.append(f"{unrated} reviews have no supplied star rating; they remain included in text and sentiment analysis.")
        cat=df.groupby("category").sentiment.apply(lambda s:(s=="negative").mean()*100).sort_values(ascending=False)
        if len(cat): findings.append(f"{cat.index[0]} has the highest negative sentiment share by category ({cat.iloc[0]:.1f}%).")
    return findings or ["Add valid reviews to calculate findings."]
