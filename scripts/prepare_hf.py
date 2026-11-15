"""Siapkan file untuk deploy ke Hugging Face (tidak mengunggah apa pun).

1. Menulis model card  -> models/indobert-smsa/README.md
2. Menyusun folder Space -> dist/hf-space/

Contoh:
    .venv/Scripts/python.exe scripts/prepare_hf.py --model-repo USERNAME/indobert-smsa-sentiment
Langkah upload ada di DEPLOY.md.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "deploy" / "hf_space"
SPACE_DIR = ROOT / "dist" / "hf-space"
MODEL_DIR = ROOT / "models" / "indobert-smsa"

SPACE_FILES = [
    "app/streamlit_app.py",
    "src/__init__.py",
    "src/data.py",
    "src/baseline.py",
    "src/indobert.py",
    "models/baseline_tfidf_lr.joblib",
    "reports/baseline_metrics.json",
    "reports/indobert_metrics.json",
]

parser = argparse.ArgumentParser()
parser.add_argument("--model-repo", required=True, help="repo model di Hub, mis. USERNAME/indobert-smsa-sentiment")
args = parser.parse_args()
if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.model_repo):
    parser.error("--model-repo harus berformat USERNAME/NAMA-REPO")

missing = [f for f in SPACE_FILES if not (ROOT / f).exists()]
if missing or not (MODEL_DIR / "model.safetensors").exists():
    raise SystemExit(f"File belum ada (jalankan training dulu): {missing or [str(MODEL_DIR)]}")

base = json.loads((ROOT / "reports/baseline_metrics.json").read_text(encoding="utf-8"))
bert = json.loads((ROOT / "reports/indobert_metrics.json").read_text(encoding="utf-8"))


def pct(x):
    return f"{x * 100:.1f}%"


values = {
    "__INDOBERT_MODEL__": args.model_repo,
    "__BASE_ACC__": pct(base["test"]["accuracy"]),
    "__BASE_F1__": pct(base["test"]["macro_f1"]),
    "__BASE_NEU__": pct(base["test"]["report"]["neutral"]["f1-score"]),
    "__BASE_MS__": f"{base['inference_ms_per_sample']:.2f} ms",
    "__BERT_ACC__": pct(bert["test"]["accuracy"]),
    "__BERT_F1__": pct(bert["test"]["macro_f1"]),
    "__BERT_NEU__": pct(bert["test"]["report"]["neutral"]["f1-score"]),
    "__BERT_MS__": f"{bert['inference_ms_per_sample']:.2f} ms",
}


def fill(text: str) -> str:
    for key, val in values.items():
        text = text.replace(key, val)
    return text


# --- 1. Model card ---
hp = bert["hyperparams"]
per_class = "\n".join(
    f"| {label} | {bert['test']['report'][label]['precision']:.3f} | {bert['test']['report'][label]['recall']:.3f} "
    f"| {bert['test']['report'][label]['f1-score']:.3f} |"
    for label in ("negative", "neutral", "positive")
)
model_card = f"""---
language: id
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: indobenchmark/indobert-base-p1
datasets:
- indonlp/indonlu
tags:
- sentiment-analysis
- indonesian
- indobert
metrics:
- accuracy
- f1
---

# IndoBERT untuk Analisis Sentimen Review Bahasa Indonesia

`indobenchmark/indobert-base-p1` yang di-fine-tune pada **SmSA** (IndoNLU) untuk klasifikasi
sentimen 3 kelas: `negative`, `neutral`, `positive`.

## Hasil (split test SmSA, 500 data)

| Model | Akurasi | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | {values['__BASE_ACC__']} | {values['__BASE_F1__']} |
| **Model ini** | **{values['__BERT_ACC__']}** | **{values['__BERT_F1__']}** |

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
{per_class}

## Cara pakai

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{args.model_repo}")
clf("pengirimannya cepat, packing rapi, barang original. mantap!")
```

## Training

- Data: SmSA train (11.000) — model terbaik dipilih dengan macro-F1 di split valid (1.260)
- Epoch: {hp['epochs']}, learning rate: {hp['lr']}, batch size: {hp['batch_size']}, max length: {hp['max_length']}, seed: {hp['seed']}
- Optimizer AdamW, linear schedule dengan warmup 10%, fp16
"""
(MODEL_DIR / "README.md").write_text(model_card, encoding="utf-8")
print(f"Model card ditulis ke {(MODEL_DIR / 'README.md').relative_to(ROOT)}")

# --- 2. Folder Space ---
if SPACE_DIR.exists():
    shutil.rmtree(SPACE_DIR)
for rel in SPACE_FILES:
    dst = SPACE_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)
for name in ("Dockerfile", "requirements.txt", "README.md"):
    text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
    (SPACE_DIR / name).write_text(fill(text), encoding="utf-8")

leftover = [p for p in SPACE_DIR.rglob("*") if p.is_file() and "__" in p.read_text(encoding="utf-8", errors="ignore")
            and re.search(r"__[A-Z_]+__", p.read_text(encoding="utf-8", errors="ignore"))]
if leftover:
    raise SystemExit(f"Placeholder belum terisi di: {leftover}")

size_mb = sum(p.stat().st_size for p in SPACE_DIR.rglob("*") if p.is_file()) / 1e6
print(f"Folder Space disusun di {SPACE_DIR.relative_to(ROOT)} ({size_mb:.1f} MB):")
for p in sorted(SPACE_DIR.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(SPACE_DIR).as_posix())
