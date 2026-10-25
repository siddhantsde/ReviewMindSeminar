"""Local VADER sentiment scoring."""
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None
_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

def analyze_sentiment(text: str) -> dict:
    """Return label, compound score, and component scores."""
    if _ANALYZER:
        scores = _ANALYZER.polarity_scores(str(text))
        compound = scores["compound"]
    else:
        # Lightweight offline fallback for installs missing vaderSentiment.
        import re
        words = set(re.findall(r"[a-z']+", str(text).lower()))
        pos = {"great", "excellent", "love", "good", "comfortable", "clear", "easy", "recommend", "reliable", "well"}
        neg = {"bad", "terrible", "poor", "broken", "fails", "issue", "drains", "disappointed", "disappointing", "confusing", "damaged", "slow", "cheaper", "stopped", "working"}
        compound = max(-1.0, min(1.0, (len(words & pos)-len(words & neg))/3))
        scores = {"pos": max(compound, 0), "neg": max(-compound, 0), "neu": 1-abs(compound), "compound": compound}
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return {**scores, "sentiment": label}

def add_sentiment(df):
    import pandas as pd
    if df.empty:
        return df.assign(sentiment=pd.Series(dtype=str), compound=pd.Series(dtype=float))
    scores = pd.DataFrame([analyze_sentiment(t) for t in df.review_text], index=df.index)
    return df.join(scores)
