import pandas as pd
from src.data_loader import load_reviews
from src.preprocessing import clean_reviews
from src.sentiment import analyze_sentiment, add_sentiment
from src.analytics import summary_metrics, product_metrics
from src.topic_analysis import discover_topics
from src.database import initialize, save_analysis, list_analyses, get_analysis
from src.report_generator import make_html_report
from src.llm_service import generate_report

def frame():
    return pd.DataFrame({"product_name":["A","A","B"],"rating":[5,1,3],"review_text":["Excellent, I love it","Terrible and broken","It is okay"],"category":["X"]*3,"review_date":[None]*3,"verified_purchase":[True]*3,"helpful_votes":[0]*3})

def test_required_columns():
    try: load_reviews(__import__("io").StringIO("rating,review_text\n5,good\n"))
    except ValueError as e: assert "product_name" in str(e)
    else: assert False

def test_missing_text_invalid_rating_and_duplicates():
    d=pd.DataFrame({"product_name":["A","A","A","A","A"],"rating":[5,"bad",4,4,2],"review_text":["ok",None,"repeat","repeat",None]})
    good,bad=clean_reviews(d); assert len(good)==3 and len(bad)==2 and good.is_duplicate.sum()==1

def test_blank_rating_is_not_invented_or_dropped():
    d=pd.DataFrame({"product_name":["A"],"rating":[None],"review_text":["Battery lasts a long time"]})
    good,bad=clean_reviews(d); assert len(good)==1 and len(bad)==0 and pd.isna(good.iloc[0].rating)
    analyzed=add_sentiment(good); assert summary_metrics(analyzed)["review_count"]==1

def test_sentiment_classification():
    assert analyze_sentiment("This is excellent and I love it")["sentiment"]=="positive"
    assert analyze_sentiment("Terrible, broken and disappointing")["sentiment"]=="negative"

def test_summary_and_product_aggregation():
    d=add_sentiment(clean_reviews(frame())[0]); m=summary_metrics(d); assert m["review_count"]==3 and m["product_count"]==2
    assert set(product_metrics(d).product_name)=={"A","B"}

def test_empty_and_small_datasets():
    empty=add_sentiment(pd.DataFrame(columns=["review_text","rating","product_name","category","review_date","verified_purchase","helpful_votes"]))
    assert summary_metrics(empty)["review_count"]==0 and len(product_metrics(empty))==0
    assert discover_topics(["same"]*3).empty

def test_topics_sufficient():
    texts=["battery lasts long and sounds clear","battery drains too soon","sound quality is crisp","battery life is disappointing","clear sound and good comfort","long battery performance"]
    assert len(discover_topics(texts,n_topics=2))>0

def test_sqlite_persistence_and_report(tmp_path):
    db=tmp_path/"test.db"; initialize(db); aid=save_analysis("demo",3,{"review_count":3},"Hello","rule_based",db)
    assert get_analysis(aid,db)["report"]=="Hello" and len(list_analyses(db))==1
    html=make_html_report("demo",{"review_count":3},["finding"],[],[],"<unsafe>","rule_based")
    assert "&lt;unsafe&gt;" in html and "finding" in html

def test_llm_unavailable_fallback(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL",raising=False)
    report,method=generate_report({"metrics":{"review_count":4},"findings":["Measured finding"]})
    assert method=="rule_based" and "Rule-based insights" in report and "Measured finding" in report
