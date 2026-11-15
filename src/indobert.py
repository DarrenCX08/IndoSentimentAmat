"""Fine-tuning & inferensi IndoBERT untuk sentimen SmSA."""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data import ID2LABEL, LABEL2ID

BASE_MODEL = "indobenchmark/indobert-base-p1"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "indobert-smsa"
# Folder lokal atau repo id Hugging Face Hub (mis. "username/indobert-smsa-sentiment").
MODEL_SOURCE = os.getenv("INDOBERT_MODEL") or str(MODEL_DIR)
MAX_LENGTH = 128


class SmsaTorchDataset(Dataset):
    """Teks yang sudah di-tokenize + label_id, untuk dipakai Trainer."""

    def __init__(self, texts, label_ids, tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(label_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_base_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, id2label=ID2LABEL, label2id=LABEL2ID
    )
    return tokenizer, model


class IndoBertSentiment:
    """Pembungkus inferensi: keluaran sama dengan src.baseline.predict."""

    def __init__(self, model_dir: str | Path = MODEL_SOURCE, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        id2label = self.model.config.id2label
        results = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i:i + batch_size], truncation=True, max_length=MAX_LENGTH,
                padding=True, return_tensors="pt",
            ).to(self.device)
            probas = torch.softmax(self.model(**batch).logits.float(), dim=-1).cpu()
            for p in probas:
                best = int(p.argmax())
                results.append({
                    "label": id2label[best],
                    "confidence": float(p[best]),
                    "probabilities": {id2label[j]: float(v) for j, v in enumerate(p)},
                })
        return results
