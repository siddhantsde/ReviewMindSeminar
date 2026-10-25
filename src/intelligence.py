"""Evidence-linked customer issue, trend, and scenario analytics.

Taxonomy matches are transparent keyword signals, not a complete ontology. The
priority score is a configurable heuristic, not a validated risk model.
"""
from __future__ import annotations

import re
from typing import Mapping

import numpy as np
import pandas as pd

# Broad enough to help discover recognizable patterns, but an issue is emitted
# only when the data contains matching evidence. Unknown issues remain visible
# through the unsupervised topic-discovery page.
ISSUE_TAXONOMY: dict[str, tuple[str, ...]] = {
    "Battery & charging": ("battery", "charge", "charging", "charger", "drains", "battery life"),
    "Performance & speed": ("slow", "lag", "laggy", "performance", "speed", "responsive", "freeze"),
    "Durability & reliability": ("durable", "durability", "broken", "broke", "stopped working", "reliable", "reliability", "defect"),
    "Sound & image quality": ("sound", "audio", "bass", "microphone", "picture", "image quality", "display", "screen"),
    "Comfort & usability": ("comfortable", "comfort", "fit", "ergonomic", "controls", "confusing", "usability"),
    "Price & value": ("price", "value", "expensive", "cheap", "cost", "overpriced", "worth"),
    "Packaging & delivery": ("delivery", "shipping", "packaging", "arrived", "package", "late"),
    "Customer service": ("support", "customer service", "refund", "replacement", "response", "representative"),
    "Connectivity": ("connect", "connectivity", "bluetooth", "pairing", "wifi", "wi-fi", "disconnect"),
    "Design & materials": ("design", "material", "plastic", "build quality", "finish", "looks"),
}

DEFAULT_WEIGHTS = {"frequency": .30, "negative_sentiment": .25, "rating_impact": .20, "recent_trend": .15, "product_coverage": .10}
EMPTY_ISSUES = ["issue", "method", "review_count", "eligible_reviews", "share_pct", "average_rating", "positive_pct", "negative_pct", "average_sentiment", "product_count", "priority_score", "priority_tier", "frequency_component", "negative_component", "rating_impact_component", "trend_component", "product_coverage_component", "trend_available", "positive_evidence", "negative_evidence", "next_action"]


def _text_mask(texts: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?!\w)", re.I)
    return texts.fillna("").astype(str).str.contains(pattern, na=False)


def _trend_component(subset: pd.DataFrame, min_period_reviews: int = 5, all_reviews: pd.DataFrame | None = None) -> tuple[float | None, dict | None]:
    if "review_date" not in subset:
        return None, None
    dated = subset.dropna(subset=["review_date"]).copy()
    all_dated = (all_reviews if all_reviews is not None else subset).dropna(subset=["review_date"]).copy()
    if dated.empty or all_dated.empty:
        return None, None
    dated["_period"] = pd.to_datetime(dated.review_date, errors="coerce").dt.to_period("M")
    all_dated["_period"] = pd.to_datetime(all_dated.review_date, errors="coerce").dt.to_period("M")
    dated = dated.dropna(subset=["_period"]); all_dated = all_dated.dropna(subset=["_period"])
    if dated.empty or all_dated.empty:
        return None, None
    periods = sorted(all_dated._period.unique())
    window = 3
    if len(periods) < window * 2:
        return None, None
    previous, recent = periods[-window*2:-window], periods[-window:]
    a, b = dated[dated._period.isin(previous)], dated[dated._period.isin(recent)]
    if min(len(a), len(b)) < min_period_reviews:
        return None, None
    prev_rate, current_rate = (a.sentiment == "negative").mean(), (b.sentiment == "negative").mean()
    # Map a bounded recent change in negative share to 0..100. No direction is
    # hidden: raw percentage-point change is returned with the component.
    pp_change = float((current_rate - prev_rate) * 100)
    previous_base, current_base = all_dated[all_dated._period.isin(previous)], all_dated[all_dated._period.isin(recent)]
    previous_share, current_share = 100*len(a)/len(previous_base), 100*len(b)/len(current_base)
    mention_pp_change = current_share - previous_share
    trend_score = float(np.clip(50 + (pp_change + mention_pp_change) / 2, 0, 100))
    return trend_score, {"previous_period": f"{previous[0]}–{previous[-1]}", "current_period": f"{recent[0]}–{recent[-1]}", "previous_negative_pct": round(prev_rate * 100, 1), "current_negative_pct": round(current_rate * 100, 1), "negative_pp_change": round(pp_change, 1), "previous_mentions": len(a), "current_mentions": len(b), "previous_issue_share_pct": round(previous_share, 1), "current_issue_share_pct": round(current_share, 1), "issue_share_pp_change": round(mention_pp_change, 1), "previous_n": len(a), "current_n": len(b)}


def pain_points(df: pd.DataFrame, *, min_reviews: int = 3, weights: Mapping[str, float] | None = None, min_period_reviews: int = 5) -> pd.DataFrame:
    """Return taxonomy-derived issues with explicit eligible denominator and evidence."""
    if df.empty or "review_text" not in df:
        return pd.DataFrame(columns=EMPTY_ISSUES)
    min_reviews = max(1, int(min_reviews))
    eligible = df[df.review_text.fillna("").astype(str).str.strip().ne("")].copy()
    if eligible.empty:
        return pd.DataFrame(columns=EMPTY_ISSUES)
    weights = dict(weights or DEFAULT_WEIGHTS)
    required = set(DEFAULT_WEIGHTS)
    if set(weights) != required or any(not np.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Weights must be non-negative values for all five components with a positive total.")
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    baseline_rating = pd.to_numeric(eligible.get("rating", pd.Series(index=eligible.index, dtype=float)), errors="coerce").mean()
    product_total = max(1, eligible.get("product_name", pd.Series("Unknown", index=eligible.index)).nunique())
    rows = []
    texts = eligible.review_text
    for issue, terms in ISSUE_TAXONOMY.items():
        mask = _text_mask(texts, terms)
        group = eligible.loc[mask].copy()
        n = len(group)
        if n < min_reviews:
            continue
        sentiment = group.get("sentiment", pd.Series("neutral", index=group.index)).fillna("neutral")
        neg = float((sentiment == "negative").mean() * 100)
        rating = pd.to_numeric(group.get("rating", pd.Series(index=group.index, dtype=float)), errors="coerce").mean()
        # Rating-impact score measures a theme's lower-than-baseline rating gap.
        impact = float(np.clip(((baseline_rating - rating) / 4) * 100, 0, 100)) if pd.notna(baseline_rating) and pd.notna(rating) else None
        trend_component, trend = _trend_component(group, min_period_reviews, eligible)
        products = group.get("product_name", pd.Series("Unknown", index=group.index)).nunique()
        components = {
            "frequency": min(100., 100 * n / len(eligible)),
            "negative_sentiment": neg,
            "rating_impact": impact,
            "recent_trend": trend_component,
            "product_coverage": 100 * products / product_total,
        }
        active_weights = {k: w for k, w in weights.items() if components[k] is not None}
        divisor = sum(active_weights.values()) or 1
        score = sum(float(components[k]) * w for k, w in active_weights.items()) / divisor
        tier = "High" if score >= 65 else "Elevated" if score >= 40 else "Monitor"
        positives = group[sentiment == "positive"].review_text.head(2).astype(str).tolist()
        negatives = group[sentiment == "negative"].review_text.head(3).astype(str).tolist()
        action = f"Review the {len(negatives)} negative example(s) and validate this pattern with product or support records." if negatives else "Inspect representative mentions to determine whether this theme needs product follow-up."
        rows.append({"issue": issue, "method": "Transparent keyword taxonomy", "review_count": n, "eligible_reviews": len(eligible), "share_pct": round(100*n/len(eligible),1), "average_rating": round(float(rating),2) if pd.notna(rating) else None, "positive_pct": round(float((sentiment == "positive").mean()*100),1), "negative_pct": round(neg,1), "average_sentiment": round(float(pd.to_numeric(group.get("compound", pd.Series(index=group.index, dtype=float)), errors="coerce").mean()),3) if "compound" in group and group.compound.notna().any() else None, "product_count": int(products), "priority_score": round(score,1), "priority_tier": tier, "frequency_component": round(components["frequency"],1), "negative_component": round(neg,1), "rating_impact_component": round(impact,1) if impact is not None else None, "trend_component": round(trend_component,1) if trend_component is not None else None, "product_coverage_component": round(components["product_coverage"],1), "trend_available": trend is not None, "trend": trend, "positive_evidence": positives, "negative_evidence": negatives, "next_action": action, "review_indices": group.index.tolist()})
    if not rows:
        return pd.DataFrame(columns=EMPTY_ISSUES)
    return pd.DataFrame(rows).sort_values(["priority_score", "review_count"], ascending=False).reset_index(drop=True)


def hidden_problems(df: pd.DataFrame, issues: pd.DataFrame | None = None, *, high_rating: float = 4.0, issue_negative_pct: float = 45.0, min_issue_reviews: int = 3) -> pd.DataFrame:
    """Flag potential investigation candidates without calling them defects."""
    cols = ["product_name", "average_rating", "product_reviews", "issue", "issue_reviews", "issue_negative_pct", "issue_share_pct", "reason", "evidence"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    issues = issues if issues is not None else pain_points(df, min_reviews=min_issue_reviews)
    records = []
    if issues.empty:
        return pd.DataFrame(columns=cols)
    for _, issue in issues.iterrows():
        matched = df.loc[df.index.isin(issue.review_indices)]
        for name, group in matched.groupby("product_name"):
            n = len(group)
            if n < min_issue_reviews:
                continue
            negative = float((group.sentiment == "negative").mean() * 100)
            product_rows = df[df.product_name == name]
            avg = pd.to_numeric(product_rows.rating, errors="coerce").mean()
            if pd.notna(avg) and avg >= high_rating and negative >= issue_negative_pct:
                evidence = group[group.sentiment == "negative"].review_text.head(2).astype(str).tolist()
                records.append({"product_name": name, "average_rating": round(float(avg),2), "product_reviews": len(product_rows), "issue": issue.issue, "issue_reviews": n, "issue_negative_pct": round(negative,1), "issue_share_pct": round(100*n/len(product_rows),1), "reason": f"{n} of {len(product_rows)} product reviews mention {issue.issue}; {negative:.1f}% of those mentions are negative despite a {avg:.2f}/5 overall rating.", "evidence": evidence})
    return pd.DataFrame(records, columns=cols).sort_values(["issue_negative_pct", "issue_reviews"], ascending=False).reset_index(drop=True) if records else pd.DataFrame(columns=cols)


def trend_analysis(df: pd.DataFrame, *, window_months: int = 1, min_period_reviews: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly metrics plus latest-vs-previous issue shifts with explicit sample sizes."""
    monthly_cols = ["month", "reviews", "average_rating", "negative_pct", "review_volume_rolling_3m", "volume_spike"]
    change_cols = ["issue", "previous_period", "current_period", "previous_reviews", "current_reviews", "previous_issue_share_pct", "current_issue_share_pct", "issue_share_pp_change", "previous_negative_pct", "current_negative_pct", "negative_pp_change", "negative_relative_change_pct"]
    dated = df.dropna(subset=["review_date"]).copy() if "review_date" in df else pd.DataFrame()
    if dated.empty:
        return pd.DataFrame(columns=monthly_cols), pd.DataFrame(columns=change_cols)
    dated["_month"] = pd.to_datetime(dated.review_date, errors="coerce").dt.to_period("M")
    dated = dated.dropna(subset=["_month"])
    if dated.empty:
        return pd.DataFrame(columns=monthly_cols), pd.DataFrame(columns=change_cols)
    monthly = dated.groupby("_month").agg(reviews=("review_text", "size"), average_rating=("rating", "mean"), negative_pct=("sentiment", lambda s: 100*(s=="negative").mean())).reset_index().rename(columns={"_month":"month"})
    monthly["month"] = monthly.month.astype(str)
    monthly[["average_rating","negative_pct"]] = monthly[["average_rating","negative_pct"]].round(1)
    monthly["review_volume_rolling_3m"] = monthly.reviews.rolling(3, min_periods=1).mean().round(1)
    monthly["volume_spike"] = monthly.reviews >= monthly.review_volume_rolling_3m * 1.5
    if len(monthly) < window_months * 2:
        return monthly[monthly_cols], pd.DataFrame(columns=change_cols)
    periods = sorted(dated._month.unique()); cur = periods[-window_months:]; prev = periods[-2*window_months:-window_months]
    a, b = dated[dated._month.isin(prev)], dated[dated._month.isin(cur)]
    rows=[]
    for issue, terms in ISSUE_TAXONOMY.items():
        ma, mb = _text_mask(a.review_text, terms), _text_mask(b.review_text, terms)
        ga, gb = a.loc[ma], b.loc[mb]
        if min(len(ga), len(gb)) < min_period_reviews:
            continue
        pa, pb = 100*(ga.sentiment=="negative").mean(), 100*(gb.sentiment=="negative").mean()
        issue_share_a, issue_share_b = 100*len(ga)/len(a), 100*len(gb)/len(b)
        rows.append({"issue":issue,"previous_period":f"{prev[0]}–{prev[-1]}","current_period":f"{cur[0]}–{cur[-1]}","previous_reviews":len(ga),"current_reviews":len(gb),"previous_issue_share_pct":round(issue_share_a,1),"current_issue_share_pct":round(issue_share_b,1),"issue_share_pp_change":round(issue_share_b-issue_share_a,1),"previous_negative_pct":round(pa,1),"current_negative_pct":round(pb,1),"negative_pp_change":round(pb-pa,1),"negative_relative_change_pct":round((pb-pa)/pa*100,1) if pa else None})
    changes=pd.DataFrame(rows,columns=change_cols).sort_values("negative_pp_change",ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=change_cols)
    return monthly[monthly_cols], changes


def monthly_issue_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month taxonomy mention counts and shares of dated reviews."""
    cols = ["month", "issue", "mentions", "dated_reviews", "share_pct"]
    if df.empty or "review_date" not in df:
        return pd.DataFrame(columns=cols)
    dated = df.dropna(subset=["review_date"]).copy()
    if dated.empty:
        return pd.DataFrame(columns=cols)
    dated["_month"] = pd.to_datetime(dated.review_date, errors="coerce").dt.to_period("M")
    dated = dated.dropna(subset=["_month"])
    if dated.empty:
        return pd.DataFrame(columns=cols)
    rows=[]
    for month, group in dated.groupby("_month", sort=True):
        texts=group.review_text.fillna("").astype(str)
        for issue, terms in ISSUE_TAXONOMY.items():
            count=int(_text_mask(texts, terms).sum())
            if count:
                rows.append({"month":str(month),"issue":issue,"mentions":count,"dated_reviews":len(group),"share_pct":round(100*count/len(group),1)})
    return pd.DataFrame(rows,columns=cols)


def rating_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "rating" not in df or "sentiment" not in df:
        return df.iloc[0:0].copy()
    rated = df.dropna(subset=["rating"]).copy()
    return rated[((rated.rating >= 4) & (rated.sentiment == "negative")) | ((rated.rating <= 2) & (rated.sentiment == "positive"))]


def scenario_estimate(*, selling_price: float, unit_sales: int, affected_pct: float, retention_pct: float) -> dict:
    values = (selling_price, unit_sales, affected_pct, retention_pct)
    if any(not np.isfinite(v) for v in values) or selling_price < 0 or unit_sales < 0 or not 0 <= affected_pct <= 100 or not 0 <= retention_pct <= 100:
        raise ValueError("Price and unit sales must be non-negative; percentages must be between 0 and 100.")
    affected = unit_sales * affected_pct / 100
    retained = affected * retention_pct / 100
    return {"estimated_affected_units": affected, "illustrative_retained_units": retained, "illustrative_sales_value": retained * selling_price, "formula": "unit_sales × assumed_affected_share × assumed_retention_share × assumed_selling_price", "label": "Illustrative scenario only; all inputs are user-entered assumptions, not measured impact."}

