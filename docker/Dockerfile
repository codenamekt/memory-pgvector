# syntax=docker/dockerfile:1.6
# hexus — multi-stage Dockerfile
#
# Stage 1 (deps): build a venv with runtime + test deps + the package,
#                 pre-download the MiniLM-L6-v2 model into the HF cache.
# Stage 2 (runtime): slim runtime image with venv, source, tests, the
#                    pre-downloaded model cache, and the entrypoint.
#
# Build:
#   docker build -f docker/Dockerfile -t hexus:latest .
#
# Why pre-download the model:
#   - First-run cold start drops from ~5-10s (network download) to ~1-2s
#     (just the model load into memory). Critical for the test loop.
#   - Image bloat: +90MB on the venv stage, +0 on runtime (the HF cache
#     is copied from deps to runtime, then HF_HUB_OFFLINE=1 in the
#     runtime env ensures the embedder never tries to reach the
#     network at test time).
#   - AIR_GAPPED_FRIENDLY: a production container built from this image
#     works offline as long as it was built with network access.
#
# Forked from andreab67/hermes-hexus (BSD-3-Clause)

ARG PYTHON_VERSION=3.11
ARG DEBIAN_RELEASE=bookworm
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ---------------------------------------------------------------------------
# Stage 1: deps — install the package + test deps + the BERT model
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS deps

# Re-declare ARGs inside the stage so they're in scope for ENV/RUN.
# (Global ARGs from above the first FROM aren't reliably accessible in
# later instructions within a stage.)
ARG EMBED_MODEL

# Hardcode the HF cache path here AND in the runtime stage's COPY
# source below. (Using `${HF_HOME}` as an ARG-with-default would be
# cleaner, but ARG values don't always expand reliably inside ENV
# instructions in the same stage — pinning the path is more robust.)
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY hexus/ ./hexus/
COPY tests/ ./tests/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Install CPU-only torch to avoid pulling the full CUDA stack (1+ GB
# of nvidia-* libs we don't need on a NUC with no GPU). The
# sentence-transformers MiniLM-L6-v2 model runs in well under 500MB
# of RAM on CPU and is plenty fast for the memory hot path.
#
# `.[test,mcp]` installs the test extras AND the MCP server SDK
# (mcp[cli] → FastMCP + uvicorn + starlette for the streamable-http
# transport). Both are needed for the test container because the
# tests in tests/test_mcp_server.py exercise the FastMCP wiring.
RUN pip install --upgrade pip wheel \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[test,mcp]"

# Pre-download the BERT model into HF_HOME. The runtime stage copies
# that path (which contains the hub/ subdir where huggingface_hub
# actually stores model files) into the hermes user's cache and runs
# with HF_HUB_OFFLINE=1, so the embedder finds the local copy and
# never touches the network.
#
#   git is required by huggingface_hub to fetch from a git-backed repo;
#   the model is hosted on the Hub as a git LFS repo.
RUN python -c "import os; print('HF_HOME=', os.environ.get('HF_HOME'), 'EMBED_MODEL=', os.environ.get('EMBED_MODEL', '<unset>'))" \
    && mkdir -p /root/.cache/huggingface \
    && EMBED_MODEL="$EMBED_MODEL" python -c "import os; from sentence_transformers import SentenceTransformer; m = SentenceTransformer(os.environ['EMBED_MODEL']); print('loaded:', os.environ['EMBED_MODEL'])" \
    && ls -la /root/.cache/huggingface \
    && ls -la /root/.cache/huggingface/hub 2>/dev/null \
    && du -sh /root/.cache/huggingface

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image with venv, source, tests, model cache, entrypoint
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CMD_PROFILE=shell \
    HF_HUB_OFFLINE=1

# Runtime system deps: libpq5 (psycopg), tini (PID 1), postgresql-client (psql for migration check)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        tini \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 hermes \
    && useradd -u 1000 -g 1000 -m -s /bin/bash hermes

# Copy the installed venv + the pre-downloaded HF model cache.
# The model cache must be readable by the hermes user (we copy with
# --chown since the deps stage had root ownership on the cache).
COPY --from=deps --chown=hermes:hermes /opt/venv /opt/venv
COPY --from=deps --chown=hermes:hermes /root/.cache/huggingface /home/hermes/.cache/huggingface

WORKDIR /app
COPY --chown=hermes:hermes hexus/ /app/hexus/
COPY --chown=hermes:hermes mcp_server/ /app/mcp_server/
COPY --chown=hermes:hermes tests/ /app/tests/
COPY --chown=hermes:hermes docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER hermes
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["shell"]
