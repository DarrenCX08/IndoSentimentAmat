"""Tes API. Jalankan dari root proyek: .venv/Scripts/python.exe -m pytest -q"""
import pytest
from fastapi.testclient import TestClient

from src.api import MAX_BATCH, MAX_TEXT_CHARS, app

LABELS = {"negative", "neutral", "positive"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` menjalankan lifespan -> model dimuat sekali
        yield c


def assert_valid_prediction(p):
    assert p["label"] in LABELS
    assert 0.0 <= p["confidence"] <= 1.0
    assert set(p["probabilities"]) == LABELS
    assert abs(sum(p["probabilities"].values()) - 1.0) < 1e-4
    assert p["confidence"] == pytest.approx(max(p["probabilities"].values()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] == ["baseline", "indobert"]


def test_models_info(client):
    body = client.get("/models").json()
    assert body["indobert"]["test_macro_f1"] > body["baseline"]["test_macro_f1"]


def test_predict_both_models(client):
    r = client.post("/predict", json={"text": "makanannya enak banget, pelayanannya ramah!"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"]) == {"baseline", "indobert"}
    for p in body["results"].values():
        assert_valid_prediction(p)
        assert p["label"] == "positive"
    assert set(body["latency_ms"]) == {"baseline", "indobert"}


@pytest.mark.parametrize("model", ["baseline", "indobert"])
def test_predict_single_model(client, model):
    r = client.post("/predict", json={"text": "pesanan telat dan makanannya dingin. kecewa.", "model": model})
    assert r.status_code == 200
    results = r.json()["results"]
    assert list(results) == [model]
    assert results[model]["label"] == "negative"


def test_indobert_handles_neutral(client):
    r = client.post("/predict", json={"text": "toko ini buka jam 9 pagi sampai jam 9 malam.", "model": "indobert"})
    assert r.json()["results"]["indobert"]["label"] == "neutral"


def test_batch(client):
    texts = ["bagus sekali", "jelek dan mahal", "saya membeli ini kemarin"]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [o["model"] for o in results] == ["baseline", "indobert"]
    for o in results:
        assert len(o["predictions"]) == len(texts)
        for p in o["predictions"]:
            assert_valid_prediction(p)


@pytest.mark.parametrize("payload", [
    {"text": ""},
    {"text": "   "},
    {"text": "a" * (MAX_TEXT_CHARS + 1)},
    {"text": "bagus", "model": "gpt"},
    {},
])
def test_predict_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("texts", [[], ["ok"] * (MAX_BATCH + 1), ["bagus", " "]])
def test_batch_invalid_input(client, texts):
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 422
