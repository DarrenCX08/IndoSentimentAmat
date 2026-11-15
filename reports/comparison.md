# Perbandingan Model — SmSA test set (500 data)

| Model | Akurasi | Macro-F1 | F1 negative | F1 neutral | F1 positive | Training | Inferensi/sampel |
|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 80.4% | 75.9% | 84.4% | 59.6% | 83.7% | 2.7 s | 0.13 ms |
| IndoBERT (fine-tuned) | 91.4% | 88.6% | 94.6% | 78.2% | 93.1% | 8.4 mnt | 3.36 ms |

Selisih macro-F1 (IndoBERT − baseline): **+12.7 poin**.

### Confusion matrix — TF-IDF + Logistic Regression

| asli \ prediksi | negative | neutral | positive |
|---|---|---|---|
| negative | 195 | 5 | 4 |
| neutral | 30 | 45 | 13 |
| positive | 33 | 13 | 162 |

### Confusion matrix — IndoBERT (fine-tuned)

| asli \ prediksi | negative | neutral | positive |
|---|---|---|---|
| negative | 201 | 0 | 3 |
| neutral | 14 | 61 | 13 |
| positive | 6 | 7 | 195 |
