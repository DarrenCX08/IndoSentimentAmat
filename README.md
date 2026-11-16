# IndoSentimentAmat — Analisis Sentimen Review Bahasa Indonesia

Klasifikasi sentimen (positif / netral / negatif) untuk review berbahasa Indonesia, yang
membandingkan secara terukur dua pendekatan:

1. **Baseline:** TF-IDF + Logistic Regression (scikit-learn)
2. **Transfer learning:** fine-tuning **IndoBERT** (`indobenchmark/indobert-base-p1`, Hugging Face Transformers)

Kedua model disajikan lewat **REST API FastAPI** dan **demo Streamlit**.

## Hasil

Dievaluasi pada split test resmi **SmSA** (IndoNLU, 500 review). Hyperparameter dipilih
dengan macro-F1 di split valid; split test hanya dipakai sekali.

| Model | Akurasi | Macro-F1 | F1 Negatif | F1 Netral | F1 Positif | Training | Inferensi / review |
|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 80.4% | 75.9% | 84.4% | 59.6% | 83.7% | 2.7 detik (CPU) | 0.13 ms |
| **IndoBERT** | **91.4%** | **88.6%** | **94.6%** | **78.2%** | **93.1%** | 8.4 menit (RTX 2060) | 3.36 ms (GPU) |

- IndoBERT unggul **+12,7 poin macro-F1**. Peningkatan terbesar ada di kelas **netral**
  (+18,6 poin F1), kelas paling sedikit datanya (~10% data train).
- Konsekuensinya: IndoBERT sekitar 25× lebih lambat per review, dan modelnya 476 MB
  (baseline hanya 1,9 MB).
- Detail lengkap (confusion matrix, per kelas) ada di [reports/comparison.md](reports/comparison.md).

Contoh: untuk *"barangnya sesuai deskripsi"*, baseline menebak **negatif** (49%),
sedangkan IndoBERT menebak **positif** (100%).

## Struktur proyek

```
src/
  data.py        # loader SmSA + pemetaan label
  baseline.py    # pipeline TF-IDF + LogReg, simpan/muat, predict()
  indobert.py    # dataset PyTorch, IndoBertSentiment.predict()
  api.py         # aplikasi FastAPI
app/
  streamlit_app.py   # demo interaktif
scripts/
  download_data.py   # unduh dataset SmSA
  eda.py             # eksplorasi data
  train_baseline.py  # grid search + evaluasi baseline
  train_indobert.py  # fine-tuning IndoBERT
  compare.py         # tabel perbandingan -> reports/comparison.md
  prepare_hf.py      # siapkan file deploy Hugging Face
tests/test_api.py    # tes API (pytest)
reports/             # metrik JSON, log training, perbandingan
models/              # model baseline (IndoBERT tidak di-commit, lihat bawah)
deploy/hf_space/     # template Docker Space
```

## Setup

Python 3.10. Contoh di bawah untuk Windows; di Linux/macOS ganti `.venv/Scripts/python.exe`
dengan `.venv/bin/python`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/download_data.py
```

Tanpa GPU NVIDIA, pasang PyTorch versi CPU (`--index-url https://download.pytorch.org/whl/cpu`).

## Penggunaan

```bash
# Training & evaluasi
.venv/Scripts/python.exe scripts/train_baseline.py
.venv/Scripts/python.exe scripts/train_indobert.py      # ~8 menit di RTX 2060
.venv/Scripts/python.exe scripts/compare.py

# API (dokumentasi interaktif di http://localhost:8000/docs)
.venv/Scripts/python.exe -m uvicorn src.api:app --port 8000

# Demo Streamlit (http://localhost:8501)
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py

# Tes
.venv/Scripts/python.exe -m pytest -q
```

### Contoh request API

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "pengirimannya cepat, packing rapi. mantap!", "model": "both"}'
```

Respons berisi `label`, `confidence`, dan probabilitas tiap kelas untuk masing-masing model,
beserta latensinya. Endpoint lain: `GET /health`, `GET /models`, `POST /predict/batch`.

Kalau model IndoBERT tidak tersedia, API dan demo Streamlit tetap jalan dengan baseline saja:
`"model": "both"` mengembalikan model yang dimuat (lihat `GET /health`), sedangkan
`"model": "indobert"` mengembalikan 503. Tes yang butuh IndoBERT otomatis di-skip.

## Model IndoBERT

File model hasil fine-tuning (476 MB) tidak disimpan di repo ini, karena melebihi batas
100 MB per file di GitHub. Ada dua cara mendapatkannya:
- Latih ulang dengan `scripts/train_indobert.py`, atau
- Unduh dari Hugging Face Hub (lihat [DEPLOY.md](DEPLOY.md)), lalu set env
  `INDOBERT_MODEL=USERNAME/indobert-smsa-sentiment`.

## Dataset

[SmSA](https://github.com/IndoNLP/indonlu) dari benchmark IndoNLU (Wilie et al., 2020):
11.000 data train, 1.260 data valid, dan 500 data test, berisi komentar dan review dari
platform online Indonesia.
