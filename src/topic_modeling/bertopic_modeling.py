from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TopicSummary:
    topic_id: int
    count: int
    name: str
    top_words: List[Tuple[str, float]]


def fit_bertopic(
    documents: List[str],
    *,
    nr_topics: Optional[int] = None,
    min_topic_size: int = 10,
) -> Dict[str, Any]:
    """
    Fit BERTopic on a list of documents (abstracts).
    Returns:
      - model: fitted BERTopic instance
      - topics: list[int] topic assignment per document
      - probs: probabilities (may be None depending on BERTopic settings)
      - summaries: list[TopicSummary] sorted by frequency (excluding -1 outliers)
    """
    if not documents:
        raise ValueError("No documents provided for topic modeling.")

    # Lazy imports keep cold-start lighter for non-topic workflows.
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    def _fit_with_vectorizer(min_df: int):
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df)
        model = BERTopic(
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            min_topic_size=min_topic_size,
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probs = model.fit_transform(documents)
        return model, topics, probs

    # Small result sets from "Theme explorer" can make min_df=2 invalid (sklearn raises:
    # "max_df corresponds to < documents than min_df"). Retry with min_df=1 in that case.
    try:
        model, topics, probs = _fit_with_vectorizer(min_df=2)
    except ValueError as e:
        msg = str(e)
        if "max_df corresponds to < documents than min_df" in msg or "max_df corresponds to fewer documents than min_df" in msg:
            model, topics, probs = _fit_with_vectorizer(min_df=1)
        else:
            raise

    # Build topic summaries
    info = model.get_topic_info()
    summaries: List[TopicSummary] = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        count = int(row["Count"])
        name = str(row.get("Name") or f"Topic {topic_id}")
        words = model.get_topic(topic_id) or []
        top_words = [(str(w), float(s)) for w, s in words[:10]]
        summaries.append(TopicSummary(topic_id=topic_id, count=count, name=name, top_words=top_words))

    summaries.sort(key=lambda s: s.count, reverse=True)

    return {
        "model": model,
        "topics": topics,
        "probs": probs,
        "summaries": summaries,
    }

