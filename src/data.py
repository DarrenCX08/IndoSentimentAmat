"""Loader untuk dataset SmSA (IndoNLU) - analisis sentimen review Bahasa Indonesia."""
import csv
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLITS = {"train": "train_preprocess.tsv", "valid": "valid_preprocess.tsv", "test": "test_preprocess.tsv"}
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_smsa(split: str = "train") -> pd.DataFrame:
    """Kembalikan DataFrame dengan kolom `text`, `label`, `label_id` untuk split train/valid/test."""
    if split not in SPLITS:
        raise ValueError(f"split harus salah satu dari {list(SPLITS)}, bukan {split!r}")
    df = pd.read_csv(
        RAW_DIR / SPLITS[split],
        sep="\t",
        header=None,
        names=["text", "label"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    df["label_id"] = df["label"].map(LABEL2ID)
    return df
