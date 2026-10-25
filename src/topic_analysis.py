"""Deterministic local TF-IDF/NMF topic discovery."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def discover_topic_assignments(texts, n_topics: int = 6, top_words: int = 6, min_topic_size: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Return deterministic NMF topic summaries and assignments by input row."""
    texts = [str(t) for t in texts if str(t).strip()]
    if len(texts) < 5 or len(set(texts)) < 3:
        return pd.DataFrame(columns=["topic", "keywords", "review_count"]), pd.Series(dtype="object")
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=1500, ngram_range=(1,2), min_df=2)
        matrix = vec.fit_transform(texts)
        k = min(n_topics, matrix.shape[0]-1, matrix.shape[1])
        if k < 1: return pd.DataFrame(columns=["topic", "keywords", "review_count"]), pd.Series(dtype="object")
        model = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=300)
        weights = model.fit_transform(matrix)
        terms = vec.get_feature_names_out()
        rows=[]; assignment = weights.argmax(1)
        labels = {}
        for i, comp in enumerate(model.components_):
            ix = comp.argsort()[-top_words:][::-1]
            label = f"Topic {i+1}"
            count = int((assignment == i).sum())
            if count >= max(1, int(min_topic_size)):
                rows.append({"topic": label, "keywords": ", ".join(terms[ix]), "review_count": count})
                labels[i] = label
        assignment_labels = pd.Series([labels.get(int(i), "Below minimum size") for i in assignment], dtype="object")
        result = pd.DataFrame(rows).sort_values("review_count", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["topic", "keywords", "review_count"])
        return result, assignment_labels
    except (ValueError, FloatingPointError):
        return pd.DataFrame(columns=["topic", "keywords", "review_count"]), pd.Series(dtype="object")

def discover_topics(texts, n_topics: int = 6, top_words: int = 6) -> pd.DataFrame:
    return discover_topic_assignments(texts, n_topics=n_topics, top_words=top_words)[0]
