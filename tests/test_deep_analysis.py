import pandas as pd
from src.deep_analysis import data_quality, rating_statistics, rating_distribution, aspect_analysis, monthly_trend

def test_deep_statistics_include_missing_and_rated_reviews():
    df=pd.DataFrame({"product_name":["A","A","B"],"review_text":["Battery life is great","Battery drains quickly","Comfortable fit"],"rating":[5,2,None],"sentiment":["positive","negative","positive"],"review_date":pd.to_datetime(["2025-01-02","2025-01-15",None]),"verified_purchase":[True,False,True]})
    quality=data_quality(df); stats=rating_statistics(df); dist=rating_distribution(df); aspects=aspect_analysis(df); trend=monthly_trend(df)
    assert quality["Total reviews"]==3 and quality["Unrated reviews"]==1
    assert stats["rated_count"]==2 and stats["mean"]==3.5 and stats["ci_low"] is not None
    assert int(dist.reviews.sum())==2
    battery=aspects[aspects.aspect=="Battery life"].iloc[0]
    assert battery.mentions==2 and battery.negative_pct==50.0
    assert len(trend)==1 and trend.iloc[0].reviews==2

def test_deep_statistics_handle_empty_or_unrated_data():
    empty=pd.DataFrame(columns=["product_name","review_text","rating","sentiment","review_date"])
    assert rating_statistics(empty)["mean"] is None
    assert rating_distribution(empty).reviews.sum()==0
    assert aspect_analysis(empty).empty
    assert monthly_trend(empty).empty
