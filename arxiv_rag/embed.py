"""Embedding model wrapper — cached singleton, Metal-accelerated on Apple Silicon.

Model: all-MiniLM-L6-v2
  - 22 MB download, 384-dim embeddings
  - Measured on M4 Max: 2221 texts/s on MPS vs 635 on CPU at batch=64 (3.5x)
  - At batch=1 the gap collapses to 1.13x — GPU transfer overhead dominates

Swap to "all-mpnet-base-v2" in config for ~5% better recall at ~2x slower.
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _best_device() -> str:
    """Pick the fastest available torch backend.

    MPS = Apple's Metal Performance Shaders, i.e. the integrated GPU.
    Cached because torch import + probe isn't free and the answer never changes.
    """
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _model(model_name: str, device: str):
    """Load model once and cache it.

    Without this cache every call re-reads ~90 MB from disk and re-initializes
    torch (~2-3s), turning a 30-second ingest into several minutes.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                batch_size: int = 64, show_progress: bool = False,
                device: str | None = None) -> np.ndarray:
    """Embed a list of strings. Returns (N, D) float32 array, L2-normalized.

    Args:
        texts: List of strings to embed.
        model_name: Sentence-transformers model identifier.
        batch_size: Encoding batch size (lower = less RAM, slower).
        show_progress: Show a tqdm progress bar.
        device: Force a torch device. None = auto-detect (mps/cuda/cpu).

    Returns:
        Array of shape (len(texts), embedding_dim), each row unit-length.
    """
    model = _model(model_name, device or _best_device())
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # cosine similarity == dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2",
                device: str | None = None) -> np.ndarray:
    """Embed a single query string. Returns (D,) float32 array.

    Deliberately does NOT force cpu despite MPS being only 1.13x faster at
    batch=1: _model's cache holds ONE entry, so alternating devices between
    ingest and query would evict and reload the model on every single call.
    Sharing the device is worth far more than the 13% single-query delta.
    """
    return embed_texts([query], model_name=model_name, device=device)[0]
