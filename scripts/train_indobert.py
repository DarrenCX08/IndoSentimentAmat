"""Fine-tuning IndoBERT (indobenchmark/indobert-base-p1) pada SmSA.

Model terbaik dipilih berdasarkan macro-F1 di split valid (dicek tiap epoch); split test
hanya dievaluasi sekali di akhir.

Jalankan dari root proyek dengan Python venv:
    .venv/Scripts/python.exe scripts/train_indobert.py
Uji cepat (subset kecil, 1 epoch, tidak menimpa model/metrik utama):
    .venv/Scripts/python.exe scripts/train_indobert.py --smoke-test
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from transformers import (  # noqa: E402
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import LABELS, load_smsa  # noqa: E402
from src.indobert import (  # noqa: E402
    BASE_MODEL,
    MAX_LENGTH,
    MODEL_DIR,
    IndoBertSentiment,
    SmsaTorchDataset,
    load_base_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke-test", action="store_true", help="500 data train, 1 epoch, output ke folder terpisah")
args = parser.parse_args()

set_seed(args.seed)
use_gpu = torch.cuda.is_available()
print(f"Device: {torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")

train, valid, test = load_smsa("train"), load_smsa("valid"), load_smsa("test")
model_dir, metrics_path = MODEL_DIR, ROOT / "reports" / "indobert_metrics.json"
if args.smoke_test:
    train = train.sample(500, random_state=args.seed)
    valid = valid.sample(200, random_state=args.seed)
    args.epochs = 1
    model_dir = ROOT / "models" / "_smoke_indobert"
    metrics_path = ROOT / "reports" / "_smoke_indobert_metrics.json"

tokenizer, model = load_base_model()
train_ds = SmsaTorchDataset(train["text"], train["label_id"], tokenizer)
valid_ds = SmsaTorchDataset(valid["text"], valid["label_id"], tokenizer)
test_ds = SmsaTorchDataset(test["text"], test["label_id"], tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


checkpoint_dir = ROOT / "models" / "_checkpoints"
steps_per_epoch = -(-len(train_ds) // args.batch_size)
warmup_steps = int(0.1 * steps_per_epoch * args.epochs)  # warmup 10% dari total langkah
training_args = TrainingArguments(
    output_dir=str(checkpoint_dir),
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    warmup_steps=warmup_steps,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=use_gpu,
    logging_steps=50,
    report_to="none",
    seed=args.seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

start = time.perf_counter()
trainer.train()
train_seconds = time.perf_counter() - start

trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))
print(f"\nModel terbaik disimpan ke {model_dir.relative_to(ROOT)}")


def evaluate(ds, df):
    preds = np.argmax(trainer.predict(ds).predictions, axis=-1)
    pred_labels = [LABELS[i] for i in preds]
    return {
        "accuracy": accuracy_score(df["label"], pred_labels),
        "macro_f1": f1_score(df["label"], pred_labels, average="macro"),
        "weighted_f1": f1_score(df["label"], pred_labels, average="weighted"),
        "report": classification_report(df["label"], pred_labels, labels=LABELS, output_dict=True),
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(df["label"], pred_labels, labels=LABELS).tolist()},
    }, pred_labels


valid_metrics, _ = evaluate(valid_ds, valid)
test_metrics, test_pred = evaluate(test_ds, test)

print("\n=== TEST ===")
print(classification_report(test["label"], test_pred, labels=LABELS, digits=4))
print("Confusion matrix (baris=asli, kolom=prediksi) urutan", LABELS)
for label, row in zip(LABELS, test_metrics["confusion_matrix"]["matrix"]):
    print(f"  {label:<9} {row}")

# Waktu inferensi diukur dengan model tersimpan, sama seperti saat dipakai di API.
clf = IndoBertSentiment(model_dir)
clf.predict(test["text"].tolist()[:32])  # warm-up
start = time.perf_counter()
clf.predict(test["text"].tolist())
infer_ms_per_sample = (time.perf_counter() - start) / len(test) * 1000

history = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
metrics = {
    "model": BASE_MODEL,
    "hyperparams": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_length": MAX_LENGTH, "seed": args.seed},
    "best_epoch_valid_macro_f1": trainer.state.best_metric,
    "epochs_trained": trainer.state.epoch,
    "device": torch.cuda.get_device_name(0) if use_gpu else "cpu",
    "train_seconds": train_seconds,
    "inference_ms_per_sample": infer_ms_per_sample,
    "valid": valid_metrics,
    "test": test_metrics,
    "eval_history": history,
}
metrics_path.parent.mkdir(exist_ok=True)
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Metrik disimpan ke {metrics_path.relative_to(ROOT)}")
print(f"Waktu training: {train_seconds / 60:.1f} menit | inferensi: {infer_ms_per_sample:.2f} ms/sampel")

samples = [
    "makanannya enak banget, pelayanannya juga ramah. pasti balik lagi!",
    "pesanan datang terlambat dan makanannya sudah dingin. kecewa.",
    "toko ini buka jam 9 pagi sampai jam 9 malam.",
]
print("\n=== Contoh prediksi ===")
for text, res in zip(samples, clf.predict(samples)):
    print(f"[{res['label']} {res['confidence']:.2%}] {text}")
