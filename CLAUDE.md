# CLAUDE.md — IndoSentimentAmat

Analisis sentimen review Bahasa Indonesia (3 kelas: negative / neutral / positive) pada dataset
**SmSA** (IndoNLU). Proyek ini membandingkan baseline TF-IDF + Logistic Regression dengan
fine-tuning IndoBERT (`indobenchmark/indobert-base-p1`), lalu menyajikan keduanya lewat
FastAPI dan Streamlit.

Hasil di split test: macro-F1 baseline 75,9%, IndoBERT 88,6%. Detail ada di README.md dan
reports/comparison.md.

User berkomunikasi dalam Bahasa Indonesia. Komentar kode dan teks UI juga berbahasa Indonesia.

## Environment (sudah dicek, tidak perlu dicek ulang)
- **Selalu pakai `.venv/Scripts/python.exe`.** Python global (3.10) hanya berisi torch versi
  CPU dan tidak punya transformers.
- Isi venv: torch 2.14.1+cu130, transformers 5.20.2, GPU RTX 2060 6 GB (CUDA aktif). Versi
  lengkap ada di `requirements.txt`.
- Windows dengan konsol cp1252: output emoji/Unicode akan gagal. Set `PYTHONIOENCODING=utf-8`
  atau pakai `sys.stdout.reconfigure(encoding="utf-8")`. Set juga
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1` untuk menyembunyikan peringatan cache HF.
- Docker tidak terpasang.

## Perintah umum (dari root proyek)
```bash
.venv/Scripts/python.exe scripts/download_data.py           # dataset -> data/raw/
.venv/Scripts/python.exe scripts/train_baseline.py          # ~3 detik
.venv/Scripts/python.exe scripts/train_indobert.py          # ~8 menit di GPU; --smoke-test untuk uji cepat
.venv/Scripts/python.exe scripts/compare.py                 # -> reports/comparison.md
.venv/Scripts/python.exe -m uvicorn src.api:app --port 8000
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
.venv/Scripts/python.exe -m pytest -q                       # ~20 detik, butuh model IndoBERT lokal
```
Konfigurasi preview di `.claude/launch.json`: `sentiment-api` (port 8000) dan `streamlit-demo` (port 8501).

## Tidak ada di git (lihat .gitignore)
- `data/raw/`: unduh dengan `scripts/download_data.py`.
- `models/indobert-smsa/` (476 MB, melebihi batas GitHub): latih ulang dengan
  `train_indobert.py`, atau muat dari HF Hub dengan env `INDOBERT_MODEL=<user>/<repo>`.
  Default-nya folder lokal; lihat `MODEL_SOURCE` di `src/indobert.py`.
- `.venv/`, `dist/`, `.claude/`.

## Konvensi
- **Metrik utama: macro-F1.** Data tidak seimbang (neutral hanya ~10% dari train).
  Hyperparameter dan epoch terbaik dipilih di split valid; split test hanya untuk evaluasi akhir.
- `baseline.predict()` dan `IndoBertSentiment.predict()` mengembalikan bentuk yang sama, yaitu
  list `{label, confidence, probabilities}`. API dan Streamlit bergantung pada bentuk ini.
- Urutan label ada di `src/data.py`: `LABELS = ["negative", "neutral", "positive"]` beserta
  `LABEL2ID` / `ID2LABEL`.
- Skrip training menyimpan metrik ke `reports/*_metrics.json`. `compare.py`, endpoint
  `/models`, tab Perbandingan di Streamlit, dan `prepare_hf.py` semuanya membaca file ini.
- Warna chart tetap per model: baseline `#2a78d6` (biru), IndoBERT `#eb6834` (oranye).

## Jebakan yang sudah ditemukan
- **transformers 5:** `TrainingArguments` tidak punya `warmup_ratio` (pakai `warmup_steps`,
  hitung manual). Pakai `eval_strategy` dan `Trainer(processing_class=...)`. Jangan kirim
  `num_labels` bersamaan dengan `id2label`.
- **scikit-learn dikunci di 1.7.2** supaya `models/baseline_tfidf_lr.joblib` tetap bisa dimuat.
  Kalau versinya dinaikkan, latih ulang baseline.
- **Streamlit 1.65:** pakai `width="stretch"`, karena `use_container_width` sudah usang.
- **Altair grouped bar (`xOffset`/`yOffset`):** pakai `.properties(height=alt.Step(n))`.
  Tinggi bar tetap (`mark_bar(height=...)`) membuat batang saling tumpang-tindih.
- **Library `datasets` tidak dipakai.** Loader HF `indonlp/indonlu` berbasis script dan tidak
  didukung versi baru, jadi data diambil langsung dari raw GitHub IndoNLU.
- **Model IndoBERT di CPU** butuh sekitar 100–250 ms per review; di GPU sekitar 3 ms.

## Deploy
- Docker/Gradio Space di Hugging Face butuh langganan **PRO**: akun gratis mendapat error 402
  saat membuat Space. Karena itu deploy ditunda.
- Template sudah siap di `deploy/hf_space/` dan `scripts/prepare_hf.py` (menulis model card dan
  menyusun `dist/hf-space/`). Langkahnya ada di DEPLOY.md.
- Alternatif gratis yang belum dikerjakan: Streamlit Community Cloud (dari repo GitHub, model
  diambil dari HF Hub).

## Git
- Remote `origin` = https://github.com/DarrenCX08/IndoSentimentAmat (public), branch `main`.
- `user.name` diset lokal di repo ini (`DarrenCX08`); `user.name` global kosong.
