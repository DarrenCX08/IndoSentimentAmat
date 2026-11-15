# Deploy ke Hugging Face Spaces

> **Status (Nov 2026):** langkah 4 belum bisa dijalankan dengan akun gratis. Hugging Face
> menolak membuat Docker Space di CPU basic (`402 Payment Required`): Docker dan Gradio Space
> sekarang butuh langganan PRO. Langkah 1–3 (unggah repo model) tetap bisa dilakukan dengan
> akun gratis. Alternatif gratis untuk demo: Streamlit Community Cloud (dari repo GitHub).

Demo Streamlit di-deploy sebagai **Docker Space** (CPU basic, gratis). Model IndoBERT (476 MB)
disimpan di **repo model** terpisah di Hub. Space mengunduhnya saat build. Baseline (1,9 MB)
ikut di dalam Space.

Ganti `USERNAME` dengan username Hugging Face kamu. Semua perintah dijalankan dari root proyek.

## 1. Login (sekali saja)

Buat token **Write** di https://huggingface.co/settings/tokens, lalu:

```bash
.venv/Scripts/hf.exe auth login
```

Cek:

```bash
.venv/Scripts/hf.exe auth whoami
```

## 2. Susun file deploy

```bash
.venv/Scripts/python.exe scripts/prepare_hf.py --model-repo USERNAME/indobert-smsa-sentiment
```

Perintah ini menulis model card ke `models/indobert-smsa/README.md` dan menyusun folder
`dist/hf-space/`. Belum ada yang diunggah.

## 3. Unggah model IndoBERT

```bash
.venv/Scripts/hf.exe repos create USERNAME/indobert-smsa-sentiment --type model --public
```

```bash
.venv/Scripts/hf.exe upload USERNAME/indobert-smsa-sentiment models/indobert-smsa . --exclude "training_args.bin" --commit-message "IndoBERT fine-tuned on SmSA"
```

## 4. Buat & unggah Space

```bash
.venv/Scripts/hf.exe repos create USERNAME/sentimen-review-indonesia --type space --sdk docker --public
```

```bash
.venv/Scripts/hf.exe upload USERNAME/sentimen-review-indonesia dist/hf-space . --repo-type space --commit-message "Deploy demo"
```

Build pertama makan waktu sekitar 5–10 menit, karena harus memasang PyTorch CPU dan mengunduh
model. Pantau tab **Logs** di https://huggingface.co/spaces/USERNAME/sentimen-review-indonesia.

## Update

Setelah melatih ulang atau mengubah app, ulangi langkah 2. Lalu jalankan lagi perintah `upload`
di langkah 3 (kalau modelnya berubah) dan/atau langkah 4.

## Catatan

- **Repo model harus public**, karena saat build Space mengunduh model tanpa token. Repo model
  private butuh token yang di-mount sebagai build secret di Dockerfile, dan itu belum disiapkan.
- Di CPU basic, IndoBERT butuh sekitar 100–250 ms per review (di RTX 2060 sekitar 3 ms).
  Space gratis akan tidur setelah 48 jam tanpa pengunjung, dan bangun lagi saat dibuka.
- Di Space, `SENTIMENT_API_URL` dikosongkan, jadi app langsung memakai model lokal tanpa API FastAPI.
- `scikit-learn` dikunci di 1.7.2 supaya file `baseline_tfidf_lr.joblib` bisa dibaca.
  Jangan naikkan versinya tanpa melatih ulang baseline.
