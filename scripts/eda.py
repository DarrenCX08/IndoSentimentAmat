"""EDA singkat dataset SmSA. Jalankan dari root proyek: python scripts/eda.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import LABELS, SPLITS, load_smsa  # noqa: E402

for split in SPLITS:
    df = load_smsa(split)
    n_words = df["text"].str.split().str.len()
    print(f"=== {split}: {len(df)} baris ===")
    print("Distribusi label:")
    print(df["label"].value_counts().to_string())
    print(f"NaN: text={df['text'].isna().sum()}, label_id={df['label_id'].isna().sum()}")
    print(f"Panjang teks (kata): mean={n_words.mean():.1f}, median={n_words.median():.0f}, max={n_words.max()}")
    print()

train = load_smsa("train")
print("=== Contoh per kelas (train) ===")
for label in LABELS:
    sample = train.loc[train["label"] == label, "text"].iloc[0]
    print(f"[{label}] {sample[:200]}")
