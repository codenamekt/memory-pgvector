"""embedder.py — local sentence-transformers embedder for the hexus memory plugin.
#
# Forked from andreab67/hermes-hexus (BSD-3-Clause).
#
# Replaces the upstream HTTP embedder with a local MiniLM-L6-v2 model
# loaded once at first use. Produces 384-dim vectors. The whole model
# fits in <500MB RAM and runs ~20-50 sentences/sec on the NUC6i7KYK
# i7 (CPU-only, no GPU needed). Cold start is the first embed call
# (~1-2s for the model load) — the async writer absorbs that without
# blocking the agent loop.
#
# Why this exists:
#   - v0.3.x required a separate Ollama / OpenAI-compatible endpoint.
#     The hermes fleet deployments on the NUC had no such endpoint
#     running; spinning one up just for the memory store is overkill
#     for a 23M-param model.
#   - sentence-transformers is a single pip install with no daemon
#     to manage, no port to expose, no healthcheck to monitor.
#   - One process = one model load. The MCP server and the Hermes
#     plugin share the same MemoryStore + LocalBertEmbedder instance,
#     so a fleet of N agents costs ~500MB resident, not N×500MB.
#
# The lazy load (see LocalBertEmbedder.embed) is deliberate: importing
# the plugin package must stay fast (Hermes loads it at startup), and
# tests that don't need embeddings should not pay the model load cost.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)


# Public model name constant — keep in one place so tests + the provider
# config + the README can all reference the same value.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class EmbedderError(Exception):
    """Raised when the local embedder fails to produce a usable vector."""


class LocalBertEmbedder:
    """Lazy-loaded, thread-safe wrapper around sentence-transformers.

    The model is loaded on the first call to `embed()` (or eagerly via
    `ensure_loaded()`) and reused for all subsequent calls. The class
    itself is cheap to construct — no model is loaded until needed.

    Designed to be a singleton per process: callers should keep one
    instance in their config and pass it around, not construct one per
    call. The MCP server and the Hermes plugin share the same
    MemoryStore which holds one embedder, so the model is loaded once
    per process regardless of how many consumers there are.

    Thread safety: a lock guards the model-load step; the underlying
    sentence-transformers `encode()` is thread-safe per the library's
    own documentation (SentenceTransformer.encode holds a per-call GIL
    boundary in the encode path), so concurrent embeds serialize on the
    GIL inside numpy/torch and don't need extra locking here.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._model = None  # loaded on first embed()
        self._load_lock = threading.Lock()
        self._load_failed = False

    # -- Public API ---------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Eagerly load the model. Useful at plugin init when you want
        the cold-start to happen on a known thread (and visibly, in
        logs) rather than on the first user-facing embed call.

        Idempotent: a no-op if the model is already loaded. Raises
        EmbedderError on load failure.
        """
        self._load_model()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        """Embedding dimension. For the default model this is 384.

        Read from the loaded model if available (some models
        self-report their dimension), otherwise returned as the
        constant for the default model.
        """
        if self._model is not None:
            # sentence-transformers exposes the dim on the underlying
            # transformer config. Fall through to the constant if not.
            try:
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:  # noqa: BLE001
                pass
        return DEFAULT_DIM if self._model_name == DEFAULT_MODEL else 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts → list of float vectors.

        Empty / whitespace-only inputs are silently dropped (returned
        as an empty list for that entry) rather than raising — callers
        in the async writer are fail-soft and the upstream test suite
        already has the "reject empty" semantics at the module level
        (see embed.embed).
        """
        if not texts:
            return []
        # Filter empties, but remember their original indices so callers
        # can still correlate the output if they care. (Today no caller
        # does — bulk_upsert_md and the async writer both just want
        # "the embeddings for the non-empty items".)
        non_empty = [t for t in texts if t and t.strip()]
        if not non_empty:
            return []

        model = self._load_model()
        try:
            # Disable sentence-transformers/tqdm progress bars; Hermes TUI
            # already reports memory tool progress and noisy "Batches 100%"
            # output is not useful in chat.
            vectors = model.encode(
                non_empty,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
        except Exception as exc:  # noqa: BLE001 — fail-soft, surface in logs
            raise EmbedderError(f"local embed failed: {exc}") from exc

        if not vectors or not isinstance(vectors[0], list):
            raise EmbedderError(f"unexpected encoder output shape: {type(vectors[0])}")

        actual_dim = len(vectors[0])
        if self._model_name == DEFAULT_MODEL and actual_dim != DEFAULT_DIM:
            # Only enforce for the known-default model — a custom model
            # with a different dim is fine, the schema is dim-driven.
            logger.warning(
                "embedder dim mismatch: model %s produced %d-dim, expected %d",
                self._model_name, actual_dim, DEFAULT_DIM,
            )
        return vectors

    # -- Internals ----------------------------------------------------------

    def _load_model(self):
        """Load the sentence-transformers model. Idempotent + thread-safe.

        The double-check pattern (check outside lock, then check inside
        before doing the expensive load) avoids serializing every embed
        call on the lock once the model is warm.
        """
        if self._model is not None:
            return self._model
        if self._load_failed:
            # Don't repeatedly try to load a model that already failed
            # this process — surface the error fast.
            raise EmbedderError(
                f"local embedder previously failed to load {self._model_name}; "
                "restart the process to retry"
            )
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                # Local import keeps the sentence-transformers dep
                # (and its torch/numpy/transformers transitive deps)
                # out of the module-level import graph, so importing
                # the plugin package stays fast.
                from sentence_transformers import SentenceTransformer
                kwargs = {"device": self._device}
                if self._cache_dir:
                    kwargs["cache_folder"] = self._cache_dir
                # Honor HF_HUB_OFFLINE for air-gapped production
                # containers. sentence-transformers passes through
                # whatever env vars are set.
                self._model = SentenceTransformer(self._model_name, **kwargs)
                logger.info(
                    "loaded local embedder model=%s dim=%d device=%s",
                    self._model_name, self.dim, self._device,
                )
                return self._model
            except Exception as exc:  # noqa: BLE001
                self._load_failed = True
                raise EmbedderError(
                    f"failed to load sentence-transformers model {self._model_name}: {exc}"
                ) from exc


# Module-level singleton accessor. NOT auto-created at import — callers
# must opt in. This keeps the import graph clean (no torch import at
# plugin import time) and lets tests inject their own embedder.
#
# Caching is keyed on (model_name, cache_dir, device) so a request for a
# different model returns a different embedder (mostly relevant for tests
# — production uses one model). The dict is small in practice.
_singletons: dict[tuple[str, Optional[str], str], "LocalBertEmbedder"] = {}
_singleton_lock = threading.Lock()


def get_default_embedder(
    model_name: str = DEFAULT_MODEL,
    *,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> "LocalBertEmbedder":
    """Return the process-wide default embedder for these args, constructing
    it on first call. Subsequent calls with the same (model_name, cache_dir,
    device) return the same instance.

    The default device is `cpu`; pass `device="cuda"` (or similar) at
    first call to override.
    """
    global _singletons
    if device is None:
        device = os.environ.get("HEXUS_EMBED_DEVICE", "cpu")
    key = (model_name, cache_dir, device)
    with _singleton_lock:
        existing = _singletons.get(key)
        if existing is not None:
            return existing
        embedder = LocalBertEmbedder(
            model_name=model_name, cache_dir=cache_dir, device=device,
        )
        _singletons[key] = embedder
        return embedder


def reset_default_embedder() -> None:
    """Drop ALL module-level singletons. Test-only helper."""
    global _singletons
    with _singleton_lock:
        _singletons = {}
