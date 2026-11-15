"""REST API analisis sentimen review Bahasa Indonesia.

Jalankan dari root proyek:
    .venv/Scripts/python.exe -m uvicorn src.api:app --port 8000
Dokumentasi interaktif: http://localhost:8000/docs
"""
import json
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src import baseline
from src.indobert import MODEL_DIR as INDOBERT_DIR

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MAX_TEXT_CHARS = 2000
MAX_BATCH = 64


class ModelName(str, Enum):
    baseline = "baseline"
    indobert = "indobert"
    both = "both"


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class ModelOutput(BaseModel):
    model: str
    latency_ms: float
    predictions: list[Prediction]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS,
                      examples=["makanannya enak banget, pelayanannya juga ramah!"])
    model: ModelName = ModelName.both

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text tidak boleh kosong")
        return v


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    model: ModelName = ModelName.both

    @field_validator("texts")
    @classmethod
    def check_texts(cls, v: list[str]) -> list[str]:
        for i, t in enumerate(v):
            if not t.strip():
                raise ValueError(f"texts[{i}] tidak boleh kosong")
            if len(t) > MAX_TEXT_CHARS:
                raise ValueError(f"texts[{i}] melebihi {MAX_TEXT_CHARS} karakter")
        return v


class PredictResponse(BaseModel):
    text: str
    results: dict[str, Prediction]
    latency_ms: dict[str, float]


class BatchPredictResponse(BaseModel):
    results: list[ModelOutput]


MODELS: dict = {}


def load_models() -> None:
    MODELS["baseline"] = baseline.load_model()
    if INDOBERT_DIR.exists():
        from src.indobert import IndoBertSentiment

        MODELS["indobert"] = IndoBertSentiment(INDOBERT_DIR)
        MODELS["indobert"].predict(["pemanasan model"])  # warm-up CUDA agar request pertama tidak lambat


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    yield
    MODELS.clear()


app = FastAPI(
    title="Analisis Sentimen Review Bahasa Indonesia",
    description="Membandingkan baseline TF-IDF + Logistic Regression dengan IndoBERT (fine-tuned pada SmSA).",
    version="1.0.0",
    lifespan=lifespan,
)


def _run(name: str, texts: list[str]) -> tuple[list[dict], float]:
    if name not in MODELS:
        raise HTTPException(status_code=503, detail=f"model '{name}' tidak tersedia")
    start = time.perf_counter()
    if name == "baseline":
        preds = baseline.predict(MODELS["baseline"], texts)
    else:
        preds = MODELS["indobert"].predict(texts)
    return preds, (time.perf_counter() - start) * 1000


def _selected(model: ModelName) -> list[str]:
    return ["baseline", "indobert"] if model == ModelName.both else [model.value]


@app.get("/health")
def health():
    device = MODELS["indobert"].device if "indobert" in MODELS else None
    return {"status": "ok", "models_loaded": sorted(MODELS), "indobert_device": device}


@app.get("/models")
def models_info():
    """Metrik tiap model di split test SmSA (dari reports/*_metrics.json)."""
    info = {}
    for name, file in [("baseline", "baseline_metrics.json"), ("indobert", "indobert_metrics.json")]:
        path = REPORTS_DIR / file
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        info[name] = {
            "loaded": name in MODELS,
            "test_accuracy": m["test"]["accuracy"],
            "test_macro_f1": m["test"]["macro_f1"],
            "test_f1_per_class": {k: m["test"]["report"][k]["f1-score"] for k in ("negative", "neutral", "positive")},
            "train_seconds": m["train_seconds"],
            "inference_ms_per_sample": m["inference_ms_per_sample"],
        }
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    results, latency = {}, {}
    for name in _selected(req.model):
        preds, ms = _run(name, [req.text])
        results[name], latency[name] = preds[0], round(ms, 2)
    return {"text": req.text, "results": results, "latency_ms": latency}


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    out = []
    for name in _selected(req.model):
        preds, ms = _run(name, req.texts)
        out.append({"model": name, "latency_ms": round(ms, 2), "predictions": preds})
    return {"results": out}
