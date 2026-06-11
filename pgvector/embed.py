"""embed.py — embedding client for the pgvector memory plugin.
#
# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause).
#
# Originally a module-level `embed(text, *, base_url, model)` that posted
# to an OpenAI-compatible /v1/embeddings or Ollama /api/embed endpoint
# and returned a 768-dim vector. In the memory-pgvector fork that HTTP
# path is preserved as the fallback; the default is now the local
# sentence-transformers MiniLM-L6-v2 model loaded by LocalBertEmbedder
# (see embedder.py), which produces 384-dim vectors.
#
# The dispatch in `embed()` here picks one or the other based on whether
# a `base_url` is supplied. This is fail-soft on purpose: a stale HTTP
# config that worked in v0.3.x keeps working until the operator flips
# the config key, no LLM cost in the hot path either way.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)

# Schema is vector(384) since the BERT swap. The HTTP path's dim check
# matches this so a misconfigured model (e.g. still pointing at
# nomic-embed-text, 768-dim) fails loudly at embed time rather than
# silently producing vectors that the DB will reject on insert.
EXPECTED_DIM = 384

# Cap text length to avoid pathological inputs (sentence-transformers
# truncates at the model max-seq-length, but trimming early keeps the
# log lines readable when something does go wrong).
MAX_INPUT_CHARS = 6000


class EmbeddingError(Exception):
    """Raised when the embedder fails to return a usable vector."""


def embed(
    text: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 10.0,
    expected_dim: int = EXPECTED_DIM,
) -> List[float]:
    """Return an embedding for `text`.

    Dispatch:
      - If `base_url` is None or empty: use the local sentence-transformers
        model (LocalBertEmbedder from .embedder). This is the default
        and the path the operator gets out of the box.
      - If `base_url` is set: POST to that endpoint, trying the
        OpenAI-compatible `/v1/embeddings` schema first and falling
        back to Ollama's native `/api/embed`. Preserved for operators
        who already have an embed service running.

    The `model` parameter is only used on the local path (it names the
    sentence-transformers checkpoint to load). On the HTTP path the
    model name is whatever the server is configured with — the caller
    can pass it through as documentation but it doesn't change the
    request body for the Ollama-native fallback (which omits `model`).

    Raises EmbeddingError on any failure. No retries — callers decide
    what to do with failures (we want fail-soft, not retry storms —
    that was Honcho's mistake).
    """
    if not text or not text.strip():
        raise EmbeddingError("empty input")

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    # Local BERT path (default — no base_url configured).
    if not base_url:
        # Lazy import keeps the sentence-transformers dep out of the
        # module-level import graph for operators who only use the HTTP
        # path (e.g. a CI env that just runs unit tests against a mock
        # endpoint).
        from .embedder import get_default_embedder, EmbedderError as _LocalErr, DEFAULT_MODEL
        embedder = get_default_embedder(model_name=model or DEFAULT_MODEL)
        try:
            vectors = embedder.embed([text])
        except _LocalErr as exc:
            raise EmbeddingError(str(exc)) from exc
        if not vectors:
            raise EmbeddingError("local embedder returned no vectors")
        return vectors[0]

    # HTTP fallback path (Ollama / OpenAI-compatible).
    base_url = base_url.rstrip("/")

    # Path A: OpenAI-compatible
    try:
        return _post(
            f"{base_url}/v1/embeddings",
            {"model": model or "nomic-embed-text", "input": text},
            timeout=timeout,
            expected_dim=expected_dim,
            extract=lambda d: d["data"][0]["embedding"],
        )
    except EmbeddingError as exc:
        logger.debug("OpenAI-compat embed failed (%s); trying native", exc)

    # Path B: Ollama native
    return _post(
        f"{base_url}/api/embed",
        {"model": model or "nomic-embed-text", "input": text},
        timeout=timeout,
        expected_dim=expected_dim,
        extract=lambda d: (d.get("embeddings") or [d.get("embedding")])[0],
    )


def _post(url: str, body: dict, *, timeout: float, expected_dim: int, extract) -> List[float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise EmbeddingError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(f"connection failed: {exc.reason}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise EmbeddingError(f"invalid JSON response: {exc}") from exc

    try:
        vec = extract(payload)
    except (KeyError, IndexError, TypeError) as exc:
        raise EmbeddingError(f"unexpected response shape: {exc}") from exc

    if not isinstance(vec, list) or not vec:
        raise EmbeddingError("response had no embedding array")
    if expected_dim and len(vec) != expected_dim:
        raise EmbeddingError(f"expected {expected_dim} dims, got {len(vec)}")
    return vec


def to_pgvector_literal(vec: List[float]) -> str:
    """Render a Python list of floats as a pgvector input literal.

    psycopg can also handle this via type adapters, but the literal form
    keeps the plugin dependency-light.
    """
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"
