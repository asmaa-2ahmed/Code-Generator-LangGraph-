# ── Base image ────────────────────────────────────────────────
# slim keeps the image small; 3.11 is stable and well-supported
# by all langchain / sentence-transformers wheels.
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────
# build-essential  → compiles native extensions (chromadb, tokenizers)
# curl             → optional healthcheck probe
# Cleaned up in the same layer to avoid bloating the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────
# Copy requirements first so Docker caches this layer.
# The layer is only rebuilt when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────
COPY . .

# ── Environment ───────────────────────────────────────────────
# HF_TOKEN must be supplied at runtime via:
#   docker run -e HF_TOKEN=hf_... -p 8000:8000 rag-assistant
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Port ──────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Start ─────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
