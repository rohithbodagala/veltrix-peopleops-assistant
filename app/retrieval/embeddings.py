"""Optional local CPU embeddings (fastembed, BAAI/bge-small-en-v1.5).

The model is baked into the Docker image at build time. At runtime it is loaded only
from the local cache (no network); if it is missing, retrieval falls back to BM25 only.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from app.logging_setup import log_event

logger = logging.getLogger(__name__)
MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class FastEmbedEmbedder:
    def __init__(self, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        self.name = MODEL_NAME
        self._model = TextEmbedding(MODEL_NAME, cache_dir=str(cache_dir), local_files_only=True)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.query_embed([text])), dtype=np.float32))[0]


def load_embedder(enabled: bool, cache_dir: Path) -> Embedder | None:
    if not enabled:
        log_event(logger, "embeddings_disabled")
        return None
    if not cache_dir.exists() or not any(cache_dir.rglob("*.onnx")):
        log_event(logger, "embeddings_unavailable", logging.WARNING, reason="model not in local cache", cache=str(cache_dir))
        return None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        embedder = FastEmbedEmbedder(cache_dir)
        embedder.embed_query("warm up")
        return embedder
    except Exception as exc:  # any load failure degrades to BM25-only
        log_event(logger, "embeddings_unavailable", logging.WARNING, reason=type(exc).__name__, detail=str(exc)[:200])
        return None
