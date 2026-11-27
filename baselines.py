"""Fit the two text classification baselines on the supplied training rows."""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def build_baselines(seed: int) -> dict:
    """Return independent, unfitted classifiers that both accept raw text."""
    return {
        "most_common": DummyClassifier(strategy="most_frequent"),
        "tfidf_logistic": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
    }
