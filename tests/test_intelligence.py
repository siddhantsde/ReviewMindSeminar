import pandas as pd
import pytest

from src.intelligence import (
    DEFAULT_WEIGHTS, hidden_problems, pain_points, rating_disagreements,
    scenario_estimate, trend_analysis,
)
from src.analytics import product_metrics
from src.topic_analysis import discover_topic_assignments
from src.recommendations import build_recommendations
from src.data_loader import load_sample
from src.preprocessing import clean_reviews
from src.sentiment import add_sentiment


def fixture_reviews():
    rows = []
    for i in range(12):
        rows.append({
            "product_name": "Healthy Headphones" if i < 8 else "Other",
            "rating": 5 if i < 8 else 3,
            "review_text": "Battery drains quickly during calls" if i < 6 else "Sound is clear and comfortable",
            "sentiment": "negative" if i < 6 else "positive",
            "compound": -.5 if i < 6 else .5,
            "review_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i * 35),
        })
    return pd.DataFrame(rows)


def test_pain_point_frequency_evidence_and_explicit_denominator():
    df = fixture_reviews()
    issues = pain_points(df, min_reviews=2)
    battery = issues[issues.issue == "Battery & charging"].iloc[0]
    assert battery.review_count == 6
    assert battery.eligible_reviews == len(df)
    assert battery.share_pct == 50.0
    assert battery.negative_pct == 100.0
    assert len(battery.negative_evidence) == 3
    assert set(battery.review_indices) == set(range(6))


def test_priority_weights_normalize_and_invalid_weights_fail():
    df = fixture_reviews()
    defaults = pain_points(df, min_reviews=1).iloc[0]
    assert defaults.priority_score >= 0 and defaults.priority_score <= 100
    equal = {key: 1 for key in DEFAULT_WEIGHTS}
    assert pain_points(df, min_reviews=1, weights=equal).iloc[0].priority_score >= 0
    with pytest.raises(ValueError):
        pain_points(df, weights={"frequency": 0})


def test_minimum_threshold_small_empty_and_unavailable_trends():
    df = fixture_reviews()
    assert "Battery & charging" not in set(pain_points(df, min_reviews=7).issue)
    assert pain_points(df.iloc[0:0]).empty
    no_dates = df.assign(review_date=pd.NaT)
    issue = pain_points(no_dates, min_reviews=2).iloc[0]
    assert not issue.trend_available
    monthly, changes = trend_analysis(no_dates)
    assert monthly.empty and changes.empty


def test_hidden_problem_uses_potential_signal_language_and_evidence():
    df = fixture_reviews()
    flags = hidden_problems(df, pain_points(df, min_reviews=2), high_rating=4, issue_negative_pct=80, min_issue_reviews=2)
    assert len(flags) == 1
    row = flags.iloc[0]
    assert row.product_name == "Healthy Headphones"
    assert row.issue == "Battery & charging"
    assert row.average_rating == 5
    assert row.issue_negative_pct == 100
    assert row.evidence


def test_trend_changes_have_pp_and_volume_shares_and_minimum_n():
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    df = pd.DataFrame({"review_text": ["Battery life is great" if i < 2 else "Sound is clear" if i < 20 else "Battery drains quickly" for i in range(40)],
                       "sentiment": ["negative" if i >= 20 else "positive" for i in range(40)],
                       "rating": [2 if i >= 20 else 5 for i in range(40)], "review_date": dates,
                       "product_name": "A"})
    _, changes = trend_analysis(df, window_months=1, min_period_reviews=2)
    assert "Battery & charging" in set(changes.issue)
    row = changes[changes.issue == "Battery & charging"].iloc[0]
    assert row.negative_pp_change > 0
    assert row.negative_pp_change == round(row.current_negative_pct - row.previous_negative_pct, 1)
    assert row.issue_share_pp_change > 0
    _, too_strict = trend_analysis(df, window_months=1, min_period_reviews=100)
    assert too_strict.empty


def test_rating_disagreement_scenario_and_product_comparison():
    df = fixture_reviews().copy()
    df.loc[0, ["rating", "sentiment"]] = [5, "negative"]
    df.loc[1, ["rating", "sentiment"]] = [1, "positive"]
    assert len(rating_disagreements(df)) == 6
    assert len(product_metrics(df)) == 2
    result = scenario_estimate(selling_price=100, unit_sales=1000, affected_pct=10, retention_pct=25)
    assert result["illustrative_sales_value"] == 2500
    with pytest.raises(ValueError):
        scenario_estimate(selling_price=100, unit_sales=10, affected_pct=101, retention_pct=50)


def test_topic_model_failure_and_rule_recommendation_fallback():
    topics, assignments = discover_topic_assignments(["same", "same", "same"], n_topics=6)
    assert topics.empty and assignments.empty
    recs = build_recommendations(fixture_reviews(), min_reviews=2)
    assert recs and "eligible reviews" in recs[0]["finding"]


def test_bundled_synthetic_data_demonstrates_hidden_issue_and_emerging_change():
    sample, rejected = clean_reviews(load_sample())
    analyzed = add_sentiment(sample)
    assert len(analyzed) >= 1000 and rejected.empty
    flags = hidden_problems(analyzed, pain_points(analyzed), min_issue_reviews=3)
    assert ((flags.product_name == "Auralis Pulse Headphones") & (flags.issue == "Battery & charging")).any()
    _, changes = trend_analysis(analyzed, window_months=3, min_period_reviews=3)
    battery = changes[changes.issue == "Battery & charging"].iloc[0]
    delivery = changes[changes.issue == "Packaging & delivery"].iloc[0]
    assert battery.issue_share_pp_change > 0
    assert delivery.negative_pp_change < 0

