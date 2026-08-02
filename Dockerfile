# syntax=docker/dockerfile:1

# ── Stage 1: build the virtualenv ────────────────────────────────────────────
# Kept separate so pip, build wheels and compiler caches never reach the final
# image. Only the finished .venv is copied forward.
FROM python:3.11-slim AS builder

# CPU-ONLY TORCH. This is the single biggest size decision in the file: the
# default Linux torch wheel bundles CUDA and is ~2.5 GB; the CPU wheel is
# ~200 MB. The deployed service does retrieval, not ingest, so it never needs a
# GPU — see PLAN-v3.md Phase 4 (generation runs on a hosted API, not here).
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the embedding model into the image (~90 MB). Without this the first
# request downloads it from HuggingFace — slow, and a hard failure in any
# network-restricted environment.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/huggingface /home/app/.cache/huggingface

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    # The model is baked in above. Without this, sentence-transformers still
    # calls huggingface.co on every load and burns 5 retries (~40s) before
    # falling back to cache — measured. Also removes a hard runtime dependency
    # on an external host.
    HF_HUB_OFFLINE=1 \
    OLLAMA_BASE_URL=http://ollama:11434

COPY arxiv_rag/ ./arxiv_rag/
COPY scripts/ ./scripts/
COPY web/ ./web/

# data/ is deliberately NOT copied: 110 MB of PDFs plus the Chroma DB, and a
# stateless container cannot rebuild it on boot. Mount it as a volume (compose)
# or fetch from object storage (PLAN-v3.md Phase 4).
RUN mkdir -p /app/data && chown -R app:app /app /home/app

USER app
EXPOSE 8001

# Reports honestly: /api/health returns status "empty" when no index is mounted,
# so an unhealthy container is distinguishable from a misconfigured one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8001/api/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "arxiv_rag.api:app", "--host", "0.0.0.0", "--port", "8001"]
