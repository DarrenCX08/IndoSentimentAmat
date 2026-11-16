"""Tes API. Jalankan dari root proyek: .venv/Scripts/python.exe -m pytest -q

Tes yang butuh IndoBERT di-skip otomatis kalau modelnya tidak ada (mis. clone baru tanpa
models/indobert-smsa/ dan tanpa env INDOBERT_MODEL); tes baseline tetap jalan.
"""
import pytest
from fastapi.testclient import TestClient

import src.api as api
import src.indobert as indobert
from src.api import MAX_BATCH, MAX_TEXT_CHARS, MODELS, app

LABELS = {"negative", "neutral", "positive"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` menjalankan lifespan -> model dimuat sekali
        yield c


@pytest.fixture
def needs_indobert(client):
    if "indobert" not in MODELS:
        pytest.skip("model IndoBERT tidak tersedia (latih dulu atau set env INDOBERT_MODEL)")


@pytest.fixture
def without_indobert(client):
    """Simulasikan API yang hanya memuat baseline; MODELS dipulihkan setelah tes."""
    saved = dict(MODELS)
    MODELS.pop("indobert", None)
    yield
    MODELS.clear()
    MODELS.update(saved)


def assert_valid_prediction(p):
    assert p["label"] in LABELS
    assert 0.0 <= p["confidence"] <= 1.0
    assert set(p["probabilities"]) == LABELS
    assert abs(sum(p["probabilities"].values()) - 1.0) < 1e-4
    assert p["confidence"] == pytest.approx(max(p["probabilities"].values()))


# ---------- Selalu jalan (cukup baseline) ----------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "baseline" in r.json()["models_loaded"]


def test_models_info(client):
    body = client.get("/models").json()
    assert body["indobert"]["test_macro_f1"] > body["baseline"]["test_macro_f1"]


def test_predict_baseline(client):
    r = client.post("/predict", json={"text": "pesanan telat dan makanannya dingin. kecewa.", "model": "baseline"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert list(results) == ["baseline"]
    assert_valid_prediction(results["baseline"])
    assert results["baseline"]["label"] == "negative"


def test_both_falls_back_to_baseline_without_indobert(client, without_indobert):
    r = client.post("/predict", json={"text": "makanannya enak banget, pelayanannya ramah!"})
    assert r.status_code == 200
    assert list(r.json()["results"]) == ["baseline"]

    r = client.post("/predict/batch", json={"texts": ["bagus sekali", "jelek dan mahal"]})
    assert r.status_code == 200
    assert [o["model"] for o in r.json()["results"]] == ["baseline"]


def test_explicit_indobert_returns_503_without_indobert(client, without_indobert):
    r = client.post("/predict", json={"text": "bagus", "model": "indobert"})
    assert r.status_code == 503


def test_load_models_without_indobert_configured(client, without_indobert, monkeypatch):
    monkeypatch.setattr(api, "indobert_configured", lambda: False)
    MODELS.clear()
    api.load_models()
    assert sorted(MODELS) == ["baseline"]


def test_load_models_survives_indobert_failure(client, without_indobert, monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs):
            raise OSError("repo tidak ditemukan")

    monkeypatch.setattr(api, "indobert_configured", lambda: True)
    monkeypatch.setattr(indobert, "IndoBertSentiment", Broken)
    MODELS.clear()
    api.load_models()
    assert sorted(MODELS) == ["baseline"]


def test_model_configured_respects_env(monkeypatch, tmp_path):
    monkeypatch.setattr(indobert, "MODEL_DIR", tmp_path / "tidak-ada")
    monkeypatch.delenv("INDOBERT_MODEL", raising=False)
    assert not indobert.model_configured()
    monkeypatch.setenv("INDOBERT_MODEL", "user/indobert-smsa-sentiment")
    assert indobert.model_configured()


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


# ---------- Butuh IndoBERT ----------
def test_health_with_indobert(client, needs_indobert):
    assert client.get("/health").json()["models_loaded"] == ["baseline", "indobert"]


def test_predict_both_models(client, needs_indobert):
    r = client.post("/predict", json={"text": "makanannya enak banget, pelayanannya ramah!"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"]) == {"baseline", "indobert"}
    for p in body["results"].values():
        assert_valid_prediction(p)
        assert p["label"] == "positive"
    assert set(body["latency_ms"]) == {"baseline", "indobert"}


def test_predict_indobert(client, needs_indobert):
    r = client.post("/predict", json={"text": "pesanan telat dan makanannya dingin. kecewa.", "model": "indobert"})
    assert r.status_code == 200
    assert r.json()["results"]["indobert"]["label"] == "negative"


def test_indobert_handles_neutral(client, needs_indobert):
    r = client.post("/predict", json={"text": "toko ini buka jam 9 pagi sampai jam 9 malam.", "model": "indobert"})
    assert r.json()["results"]["indobert"]["label"] == "neutral"


def test_batch(client, needs_indobert):
    texts = ["bagus sekali", "jelek dan mahal", "saya membeli ini kemarin"]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [o["model"] for o in results] == ["baseline", "indobert"]
    for o in results:
        assert len(o["predictions"]) == len(texts)
        for p in o["predictions"]:
            assert_valid_prediction(p)
