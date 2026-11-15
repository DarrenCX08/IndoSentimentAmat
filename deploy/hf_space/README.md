---
title: Sentimen Review Indonesia
emoji: 💬
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
short_description: Sentimen review Indonesia — TF-IDF+LogReg vs IndoBERT
---

# 💬 Analisis Sentimen Review Bahasa Indonesia

Demo yang membandingkan dua pendekatan klasifikasi sentimen (positif / netral / negatif)
untuk review berbahasa Indonesia:

| Model | Akurasi | Macro-F1 | F1 Netral | Inferensi / review |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | __BASE_ACC__ | __BASE_F1__ | __BASE_NEU__ | __BASE_MS__ |
| IndoBERT (fine-tuned) | __BERT_ACC__ | __BERT_F1__ | __BERT_NEU__ | __BERT_MS__ |

Dievaluasi pada split test resmi **SmSA** (IndoNLU, 500 review). Waktu inferensi diukur
saat evaluasi (IndoBERT di GPU); di Space ini model berjalan di CPU sehingga lebih lambat.

- Model IndoBERT: [__INDOBERT_MODEL__](https://huggingface.co/__INDOBERT_MODEL__)
- Base model: [indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)
- Dataset: [SmSA — IndoNLU](https://github.com/IndoNLP/indonlu)
