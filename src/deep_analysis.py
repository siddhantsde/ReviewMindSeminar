"""Additional descriptive statistics for review datasets."""
import pandas as pd

ASPECTS = {
    "Battery life": ["battery", "charge", "charging", "drains"],
    "Sound quality": ["sound", "audio", "volume", "microphone", "bass"],
    "Durability & reliability": ["durable", "durability", "broken", "stopped working", "reliable", "reliability"],
    "Comfort & fit": ["comfortable", "comfort", "fit", "heavy", "lightweight"],
    "Price & value": ["price", "value", "expensive", "cheap", "cost"],
    "Ease of use": ["easy", "setup", "controls", "instructions", "simple"],
    "Delivery & packaging": ["delivery", "shipping", "packaging", "arrived", "package"],
    "Customer support": ["support", "service", "refund", "replacement", "response"],
}

def data_quality(df: pd.DataFrame) -> dict:
    n=len(df)
    text=df.get("review_text",pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    return {
        "Total reviews": n,
        "Rated reviews": int(df.rating.notna().sum()) if "rating" in df else 0,
        "Unrated reviews": int(df.rating.isna().sum()) if "rating" in df else n,
        "Unique products": int(df.product_name.nunique()) if "product_name" in df else 0,
        "Duplicate text rows": int(text.str.lower().duplicated(keep="first").sum()),
        "Missing dates": int(df.review_date.isna().sum()) if "review_date" in df else n,
        "Verified purchases": int(df.verified_purchase.sum()) if "verified_purchase" in df else 0,
        "Text completeness": round(100*text.ne("").mean(),1) if n else 0.0,
    }

def rating_statistics(df: pd.DataFrame) -> dict:
    values=pd.to_numeric(df.get("rating",pd.Series(dtype=float)),errors="coerce").dropna()
    n=len(values)
    if not n: return {"rated_count":0,"mean":None,"median":None,"std":None,"ci_low":None,"ci_high":None}
    mean=float(values.mean()); std=float(values.std(ddof=1)) if n>1 else 0.0
    margin=1.96*std/(n**0.5) if n>1 else None
    return {"rated_count":n,"mean":round(mean,2),"median":float(values.median()),"std":round(std,2),
            "ci_low":round(max(1.0,mean-margin),2) if margin is not None else None,
            "ci_high":round(min(5.0,mean+margin),2) if margin is not None else None}

def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vals=pd.to_numeric(df.get("rating",pd.Series(dtype=float)),errors="coerce").dropna()
    counts=vals.value_counts().reindex([1,2,3,4,5],fill_value=0).rename_axis("rating").reset_index(name="reviews")
    denom=int(counts.reviews.sum())
    counts["share_pct"]=(100*counts.reviews/denom).round(1) if denom else 0.0
    return counts

def aspect_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Count explicit keyword mentions and their observed sentiment mix."""
    rows=[]
    if df.empty: return pd.DataFrame(columns=["aspect","mentions","negative_pct","average_rating","evidence_strength"])
    texts=df.review_text.fillna("").astype(str).str.lower()
    for aspect,terms in ASPECTS.items():
        mask=texts.apply(lambda text:any(term in text for term in terms))
        subset=df.loc[mask]
        if subset.empty: continue
        n=len(subset); neg=100*(subset.sentiment=="negative").mean()
        rated=subset.rating.dropna()
        rows.append({"aspect":aspect,"mentions":n,"negative_pct":round(float(neg),1),
                     "average_rating":round(float(rated.mean()),2) if len(rated) else None,
                     "evidence_strength":"Higher" if n>=30 else "Moderate" if n>=10 else "Early signal"})
    return pd.DataFrame(rows,columns=["aspect","mentions","negative_pct","average_rating","evidence_strength"]).sort_values(["negative_pct","mentions"],ascending=[False,False]).reset_index(drop=True)

def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    dated=df.dropna(subset=["review_date"]).copy()
    if dated.empty: return pd.DataFrame(columns=["month","reviews","average_rating","negative_pct"])
    dated["month"]=dated.review_date.dt.to_period("M").astype(str)
    return dated.groupby("month").agg(reviews=("review_text","size"),average_rating=("rating","mean"),negative_pct=("sentiment",lambda s:100*(s=="negative").mean())).reset_index().assign(average_rating=lambda x:x.average_rating.round(2),negative_pct=lambda x:x.negative_pct.round(1))

def sentiment_by_segment(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df or df.empty: return pd.DataFrame(columns=[column,"reviews","positive_pct","neutral_pct","negative_pct"])
    return df.groupby(column).agg(reviews=("review_text","size"),positive_pct=("sentiment",lambda s:100*(s=="positive").mean()),neutral_pct=("sentiment",lambda s:100*(s=="neutral").mean()),negative_pct=("sentiment",lambda s:100*(s=="negative").mean())).reset_index().round(1)
