"""Evidence-linked, descriptive investigation recommendations."""
from .intelligence import pain_points

def build_recommendations(df, min_reviews=3):
    results=[]
    issues=pain_points(df,min_reviews=min_reviews)
    for _, row in issues.head(5).iterrows():
        evidence=row.negative_evidence or row.positive_evidence
        confidence="Higher sample support" if row.review_count>=30 else "Moderate sample support" if row.review_count>=10 else "Early signal; validate before prioritizing"
        results.append({"finding":f"{row.issue}: {row.review_count} of {row.eligible_reviews} eligible reviews ({row.share_pct}%) mention this theme; {row.negative_pct}% of mentions are negative, versus {100*(df.sentiment=='negative').mean():.1f}% overall.","action":row.next_action,"evidence":evidence,"confidence":confidence,"priority_score":row.priority_score,"priority_tier":row.priority_tier})
    if not results and len(df): results.append({"finding":"No recognized taxonomy issue met the minimum sample threshold.","action":"Inspect discovered topics and representative low-rated reviews; the taxonomy may not cover this product category.","evidence":[],"confidence":"Insufficient recognized issue evidence"})
    return results
