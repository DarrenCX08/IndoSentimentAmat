FROM python:3.10-slim

# Hugging Face Spaces menjalankan container sebagai user dengan UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SENTIMENT_API_URL=""
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Model IndoBERT diunduh dari Hub saat build, jadi container tidak perlu mengunduh ulang tiap start.
ARG INDOBERT_MODEL=__INDOBERT_MODEL__
ENV INDOBERT_MODEL=$INDOBERT_MODEL
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['INDOBERT_MODEL'])"

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
