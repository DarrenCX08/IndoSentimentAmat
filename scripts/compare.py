"""Bandingkan baseline TF-IDF+LR dengan IndoBERT di split test -> reports/comparison.md"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # konsol Windows default cp1252

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

models = {
    "TF-IDF + Logistic Regression": json.loads((REPORTS / "baseline_metrics.json").read_text(encoding="utf-8")),
    "IndoBERT (fine-tuned)": json.loads((REPORTS / "indobert_metrics.json").read_text(encoding="utf-8")),
}
labels = ["negative", "neutral", "positive"]


def pct(x):
    return f"{x * 100:.1f}%"


lines = [
    "# Perbandingan Model — SmSA test set (500 data)",
    "",
    "| Model | Akurasi | Macro-F1 | " + " | ".join(f"F1 {l}" for l in labels) + " | Training | Inferensi/sampel |",
    "|---|---|---|" + "---|" * len(labels) + "---|---|",
]
for name, m in models.items():
    t = m["test"]
    per_class = " | ".join(pct(t["report"][l]["f1-score"]) for l in labels)
    train_time = f"{m['train_seconds']:.1f} s" if m["train_seconds"] < 120 else f"{m['train_seconds'] / 60:.1f} mnt"
    lines.append(
        f"| {name} | {pct(t['accuracy'])} | {pct(t['macro_f1'])} | {per_class} | {train_time} | {m['inference_ms_per_sample']:.2f} ms |"
    )

(base, bert) = models.values()
delta = (bert["test"]["macro_f1"] - base["test"]["macro_f1"]) * 100
lines += ["", f"Selisih macro-F1 (IndoBERT − baseline): **{delta:+.1f} poin**."]

for name, m in models.items():
    lines += ["", f"### Confusion matrix — {name}", "", "| asli \\ prediksi | " + " | ".join(labels) + " |", "|---|---|---|---|"]
    for label, row in zip(labels, m["test"]["confusion_matrix"]["matrix"]):
        lines.append(f"| {label} | " + " | ".join(map(str, row)) + " |")

out = REPORTS / "comparison.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nDisimpan ke {out.relative_to(ROOT)}")
