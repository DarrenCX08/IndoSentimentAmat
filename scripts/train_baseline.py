"""Latih & evaluasi baseline TF-IDF + Logistic Regression.

Hyperparameter dipilih berdasarkan macro-F1 di split valid; split test hanya dipakai sekali
di akhir. Jalankan dari root proyek: python scripts/train_baseline.py
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

from src.baseline import MODEL_PATH, build_pipeline, predict, save_model  # noqa: E402
from src.data import LABELS, load_smsa  # noqa: E402

train, valid, test = load_smsa("train"), load_smsa("valid"), load_smsa("test")

# --- 1. Pilih hyperparameter di valid ---
grid = {
    "C": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "class_weight": [None, "balanced"],
    "ngram_range": [(1, 1), (1, 2)],
}
results = []
for C, cw, ngram in product(grid["C"], grid["class_weight"], grid["ngram_range"]):
    pipe = build_pipeline(C=C, class_weight=cw, ngram_range=ngram).fit(train["text"], train["label"])
    f1 = f1_score(valid["label"], pipe.predict(valid["text"]), average="macro")
    results.append({"C": C, "class_weight": cw, "ngram_range": list(ngram), "valid_macro_f1": f1})
    print(f"C={C:<5} class_weight={str(cw):<9} ngram={ngram}  valid macro-F1={f1:.4f}")

best = max(results, key=lambda r: r["valid_macro_f1"])
print(f"\nTerbaik: {best}")

# --- 2. Latih ulang dengan config terbaik, evaluasi di valid & test ---
start = time.perf_counter()
pipe = build_pipeline(
    C=best["C"], class_weight=best["class_weight"], ngram_range=tuple(best["ngram_range"])
).fit(train["text"], train["label"])
train_seconds = time.perf_counter() - start


def evaluate(df):
    pred = pipe.predict(df["text"])
    return {
        "accuracy": accuracy_score(df["label"], pred),
        "macro_f1": f1_score(df["label"], pred, average="macro"),
        "weighted_f1": f1_score(df["label"], pred, average="weighted"),
        "report": classification_report(df["label"], pred, labels=LABELS, output_dict=True),
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(df["label"], pred, labels=LABELS).tolist()},
    }


start = time.perf_counter()
test_metrics = evaluate(test)
infer_ms_per_sample = (time.perf_counter() - start) / len(test) * 1000
valid_metrics = evaluate(valid)

print("\n=== TEST ===")
print(classification_report(test["label"], pipe.predict(test["text"]), labels=LABELS, digits=4))
print("Confusion matrix (baris=asli, kolom=prediksi) urutan", LABELS)
for label, row in zip(LABELS, test_metrics["confusion_matrix"]["matrix"]):
    print(f"  {label:<9} {row}")

# --- 3. Simpan model & metrik ---
save_model(pipe)
metrics = {
    "model": "tfidf_logreg",
    "best_params": best,
    "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
    "train_seconds": train_seconds,
    "inference_ms_per_sample": infer_ms_per_sample,
    "valid": valid_metrics,
    "test": test_metrics,
    "grid_search": results,
}
reports_dir = ROOT / "reports"
reports_dir.mkdir(exist_ok=True)
(reports_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print(f"\nModel disimpan ke {MODEL_PATH.relative_to(ROOT)}")
print(f"Metrik disimpan ke reports/baseline_metrics.json")
print(f"Waktu training: {train_seconds:.2f}s | inferensi: {infer_ms_per_sample:.3f} ms/sampel")

# --- 4. Contoh prediksi ---
samples = [
    "makanannya enak banget, pelayanannya juga ramah. pasti balik lagi!",
    "pesanan datang terlambat dan makanannya sudah dingin. kecewa.",
    "toko ini buka jam 9 pagi sampai jam 9 malam.",
]
print("\n=== Contoh prediksi ===")
for text, res in zip(samples, predict(pipe, samples)):
    print(f"[{res['label']} {res['confidence']:.2%}] {text}")
