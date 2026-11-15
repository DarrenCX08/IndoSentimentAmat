"""Baseline TF-IDF + Logistic Regression untuk sentimen SmSA."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_tfidf_lr.joblib"


def build_pipeline(C: float = 1.0, class_weight=None, ngram_range=(1, 2)) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=42,
        )),
    ])


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict(pipeline: Pipeline, texts: list[str]) -> list[dict]:
    """Kembalikan label, confidence, dan probabilitas semua kelas untuk tiap teks."""
    probas = pipeline.predict_proba(texts)
    classes = pipeline.classes_
    results = []
    for p in probas:
        best = p.argmax()
        results.append({
            "label": str(classes[best]),
            "confidence": float(p[best]),
            "probabilities": {str(c): float(v) for c, v in zip(classes, p)},
        })
    return results
