"""Demo Streamlit: analisis sentimen review Bahasa Indonesia (baseline vs IndoBERT).

Memanggil API FastAPI (env SENTIMENT_API_URL, default http://localhost:8000). Jika API tidak
bisa dihubungi, model dimuat langsung di proses ini — jadi app juga bisa jalan sendiri
(mis. di Hugging Face Spaces).

Jalankan dari root proyek:
    .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
"""
import json
import os
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Kosongkan (SENTIMENT_API_URL="") untuk selalu memakai model lokal, mis. di Hugging Face Spaces.
API_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000").rstrip("/")
REPORTS = ROOT / "reports"
LABELS = ["negative", "neutral", "positive"]
LABEL_ID = {"negative": "😞 Negatif", "neutral": "😐 Netral", "positive": "😊 Positif"}
MODEL_NAMES = {"baseline": "TF-IDF + LogReg", "indobert": "IndoBERT"}
# Warna kategorikal tetap per model (slot 1 & 2 palet referensi) — tidak pernah berganti.
MODEL_COLORS = {"TF-IDF + LogReg": "#2a78d6", "IndoBERT": "#eb6834"}
MAX_BATCH = 64
EXAMPLES = [
    "pengirimannya cepat, packing rapi, barang original. mantap!",
    "barangnya sesuai deskripsi",
    "toko ini buka jam 9 pagi sampai jam 9 malam.",
    "aplikasinya sering error dan lemot, tolong diperbaiki",
    "wah hebat banget, pesanan saya hilang dua kali",
]

st.set_page_config(page_title="Sentimen Review Indonesia", page_icon="💬", layout="wide")


# ---------- Backend: API dengan fallback model lokal ----------
@st.cache_data(ttl=30, show_spinner=False)
def api_available() -> bool:
    if not API_URL:
        return False
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.ok and set(r.json().get("models_loaded", [])) >= {"baseline", "indobert"}
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="Memuat model lokal…")
def local_models() -> dict:
    from src import baseline
    from src.indobert import IndoBertSentiment

    return {"baseline": baseline.load_model(), "indobert": IndoBertSentiment()}


def predict(texts: list[str]) -> dict[str, dict]:
    """-> {model: {"predictions": [...], "latency_ms": float}} untuk kedua model."""
    out = {m: {"predictions": [], "latency_ms": 0.0} for m in MODEL_NAMES}
    for i in range(0, len(texts), MAX_BATCH):
        chunk = texts[i:i + MAX_BATCH]
        if api_available():
            r = requests.post(f"{API_URL}/predict/batch", json={"texts": chunk, "model": "both"}, timeout=60)
            r.raise_for_status()
            results = {o["model"]: o for o in r.json()["results"]}
        else:
            from src import baseline

            models, results = local_models(), {}
            for name in MODEL_NAMES:
                start = time.perf_counter()
                preds = (baseline.predict(models[name], chunk) if name == "baseline"
                         else models[name].predict(chunk))
                results[name] = {"predictions": preds, "latency_ms": (time.perf_counter() - start) * 1000}
        for name in MODEL_NAMES:
            out[name]["predictions"] += results[name]["predictions"]
            out[name]["latency_ms"] += results[name]["latency_ms"]
    return out


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    files = {"baseline": "baseline_metrics.json", "indobert": "indobert_metrics.json"}
    return {m: json.loads((REPORTS / f).read_text(encoding="utf-8")) for m, f in files.items()
            if (REPORTS / f).exists()}


def model_color_scale():
    return alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))


# ---------- Header ----------
st.title("💬 Analisis Sentimen Review Bahasa Indonesia")
st.caption(
    "Membandingkan baseline **TF-IDF + Logistic Regression** dengan **IndoBERT** "
    "(`indobenchmark/indobert-base-p1`, fine-tuned pada dataset SmSA/IndoNLU)."
)
if api_available():
    backend = f"API `{API_URL}`"
else:
    backend = "model lokal" + (" (API tidak terhubung)" if API_URL else "")
st.caption(f"Backend: {backend}")

tab_single, tab_batch, tab_compare = st.tabs(["🔍 Coba Prediksi", "📄 Batch", "📊 Perbandingan Model"])

# ---------- Tab 1: satu teks ----------
with tab_single:
    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES[0]

    st.write("Contoh cepat:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex if len(ex) <= 28 else ex[:26] + "…", help=ex, width="stretch"):
            st.session_state.text = ex

    text = st.text_area("Tulis review", key="text", height=100, max_chars=2000)

    if not text.strip():
        st.info("Tulis sebuah review untuk melihat prediksinya.")
    else:
        with st.spinner("Memprediksi…"):
            res = predict([text])

        c1, c2 = st.columns(2)
        for col, name in zip((c1, c2), MODEL_NAMES):
            p = res[name]["predictions"][0]
            with col.container(border=True):
                st.markdown(f"**{MODEL_NAMES[name]}**")
                st.markdown(f"### {LABEL_ID[p['label']]}")
                st.progress(p["confidence"], text=f"Confidence {p['confidence']:.1%}")
                st.caption(f"Latensi: {res[name]['latency_ms']:.1f} ms")

        b, i = (res[m]["predictions"][0]["label"] for m in MODEL_NAMES)
        if b != i:
            st.warning(f"Kedua model **berbeda pendapat**: baseline → {LABEL_ID[b]}, IndoBERT → {LABEL_ID[i]}.")

        rows = [
            {"Model": MODEL_NAMES[m], "Kelas": LABEL_ID[k], "Probabilitas": v}
            for m in MODEL_NAMES for k, v in res[m]["predictions"][0]["probabilities"].items()
        ]
        chart = (
            alt.Chart(pd.DataFrame(rows), title="Probabilitas tiap kelas")
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("Kelas:N", sort=[LABEL_ID[l] for l in LABELS], title=None,
                        scale=alt.Scale(paddingInner=0.3)),
                yOffset=alt.YOffset("Model:N", sort=list(MODEL_COLORS), scale=alt.Scale(paddingInner=0.15)),
                x=alt.X("Probabilitas:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
                color=alt.Color("Model:N", scale=model_color_scale(), legend=alt.Legend(orient="top", title=None)),
                tooltip=["Model", "Kelas", alt.Tooltip("Probabilitas:Q", format=".1%")],
            )
            .properties(height=alt.Step(22))
        )
        st.altair_chart(chart, width="stretch")

# ---------- Tab 2: batch ----------
with tab_batch:
    st.write("Tempel beberapa review (satu per baris) atau unggah CSV dengan kolom `text`.")
    raw = st.text_area("Review (satu per baris)", height=150, value="\n".join(EXAMPLES))
    upload = st.file_uploader("…atau unggah CSV", type="csv")

    if st.button("Prediksi semua", type="primary"):
        if upload is not None:
            df_in = pd.read_csv(upload)
            if "text" not in df_in.columns:
                st.error("CSV harus punya kolom `text`.")
                st.stop()
            texts = df_in["text"].astype(str).tolist()
        else:
            texts = [t for t in raw.splitlines() if t.strip()]
        texts = [t[:2000] for t in texts if t.strip()][:1000]

        if not texts:
            st.info("Tidak ada teks untuk diprediksi.")
        else:
            with st.spinner(f"Memprediksi {len(texts)} review…"):
                res = predict(texts)
            df = pd.DataFrame({"text": texts})
            for m in MODEL_NAMES:
                df[f"{m}_label"] = [p["label"] for p in res[m]["predictions"]]
                df[f"{m}_confidence"] = [round(p["confidence"], 4) for p in res[m]["predictions"]]
            df["sepakat"] = df["baseline_label"] == df["indobert_label"]

            a, b, c = st.columns(3)
            a.metric("Jumlah review", len(df))
            b.metric("Model sepakat", f"{df['sepakat'].mean():.0%}")
            c.metric("Latensi total (base / IndoBERT)",
                     f"{res['baseline']['latency_ms']:.0f} / {res['indobert']['latency_ms']:.0f} ms")

            dist = pd.DataFrame([
                {"Model": MODEL_NAMES[m], "Kelas": LABEL_ID[l], "Jumlah": int((df[f"{m}_label"] == l).sum())}
                for m in MODEL_NAMES for l in LABELS
            ])
            st.altair_chart(
                alt.Chart(dist, title="Distribusi prediksi")
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Kelas:N", sort=[LABEL_ID[l] for l in LABELS], title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("Model:N", sort=list(MODEL_COLORS)),
                    y=alt.Y("Jumlah:Q", title="Jumlah review", axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color("Model:N", scale=model_color_scale(), legend=alt.Legend(orient="top", title=None)),
                    tooltip=["Model", "Kelas", "Jumlah"],
                )
                .properties(height=260),
                width="stretch",
            )
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("⬇️ Unduh hasil (CSV)", df.to_csv(index=False).encode("utf-8"),
                               "hasil_sentimen.csv", "text/csv")

# ---------- Tab 3: perbandingan ----------
with tab_compare:
    metrics = load_metrics()
    if len(metrics) < 2:
        st.info("Jalankan `scripts/train_baseline.py` dan `scripts/train_indobert.py` dulu untuk melihat perbandingan.")
    else:
        base, bert = metrics["baseline"]["test"], metrics["indobert"]["test"]
        st.subheader("Hasil di split test SmSA (500 review)")
        a, b, c = st.columns(3)
        a.metric("Macro-F1 IndoBERT", f"{bert['macro_f1']:.1%}", f"{(bert['macro_f1'] - base['macro_f1']) * 100:+.1f} poin vs baseline")
        b.metric("Akurasi IndoBERT", f"{bert['accuracy']:.1%}", f"{(bert['accuracy'] - base['accuracy']) * 100:+.1f} poin vs baseline")
        c.metric("Macro-F1 baseline", f"{base['macro_f1']:.1%}")

        f1_rows = [
            {"Model": MODEL_NAMES[m], "Metrik": name, "F1": val}
            for m in MODEL_NAMES
            for name, val in [("Macro-F1", metrics[m]["test"]["macro_f1"])]
            + [(LABEL_ID[l], metrics[m]["test"]["report"][l]["f1-score"]) for l in LABELS]
        ]
        order = ["Macro-F1"] + [LABEL_ID[l] for l in LABELS]
        st.altair_chart(
            alt.Chart(pd.DataFrame(f1_rows), title="F1 per kelas — kelas Netral paling diuntungkan IndoBERT")
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Metrik:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("Model:N", sort=list(MODEL_COLORS)),
                y=alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=alt.Color("Model:N", scale=model_color_scale(), legend=alt.Legend(orient="top", title=None)),
                tooltip=["Model", "Metrik", alt.Tooltip("F1:Q", format=".1%")],
            )
            .properties(height=300),
            width="stretch",
        )

        table = pd.DataFrame([
            {
                "Model": MODEL_NAMES[m],
                "Akurasi": f"{metrics[m]['test']['accuracy']:.1%}",
                "Macro-F1": f"{metrics[m]['test']['macro_f1']:.1%}",
                **{f"F1 {LABEL_ID[l]}": f"{metrics[m]['test']['report'][l]['f1-score']:.1%}" for l in LABELS},
                "Waktu training": (f"{metrics[m]['train_seconds']:.1f} detik" if metrics[m]["train_seconds"] < 120
                                   else f"{metrics[m]['train_seconds'] / 60:.1f} menit"),
                "Inferensi / review": f"{metrics[m]['inference_ms_per_sample']:.2f} ms",
            }
            for m in MODEL_NAMES
        ])
        st.dataframe(table, width="stretch", hide_index=True)

        st.subheader("Confusion matrix (baris = label asli, kolom = prediksi)")
        cols = st.columns(2)
        for col, m in zip(cols, MODEL_NAMES):
            cm = metrics[m]["test"]["confusion_matrix"]["matrix"]
            names = [LABEL_ID[l] for l in LABELS]
            col.markdown(f"**{MODEL_NAMES[m]}**")
            col.dataframe(pd.DataFrame(cm, index=names, columns=names), width="stretch")

        st.caption(
            "Trade-off: IndoBERT jauh lebih akurat, tetapi ~25× lebih lambat per review dan modelnya ~476 MB, "
            "sedangkan baseline hanya ~2 MB dan cukup dijalankan di CPU."
        )
