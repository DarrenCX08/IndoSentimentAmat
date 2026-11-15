"""Unduh dataset SmSA (IndoNLU) ke data/raw/. Jalankan dari root proyek: python scripts/download_data.py"""
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED_ROWS = {"train": 11000, "valid": 1260, "test": 500}

RAW_DIR.mkdir(parents=True, exist_ok=True)
for split, n in EXPECTED_ROWS.items():
    name = f"{split}_preprocess.tsv"
    dest = RAW_DIR / name
    urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
    rows = sum(1 for line in dest.read_text(encoding="utf-8").splitlines() if line.strip())
    status = "OK" if rows == n else f"PERINGATAN: diharapkan {n}"
    print(f"{name}: {rows} baris ({status})")
