"""Embedding model wrapper — CPU-only, cached singleton.

Model: all-MiniLM-L6-v2
  - 22 MB download, 384-dim embeddings
  - ~1000 sentences/sec on modern CPU
  - Good quality for English ML text
  - No GPU required

Swap to "all-mpnet-base-v2" in config for ~5% better recall at 2x slower.
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model(model_name: str):
    """Load model once and cache — avoids reloading on repeated calls."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed a list of strings. Returns (N, D) float32 array, L2-normalized.

    Args:
        texts: List of strings to embed.
        model_name: Sentence-transformers model identifier.
        batch_size: Batch size for encoding (lower = less RAM, lower = slower).
        show_progress: Show tqdm progress bar.

    Returns:
        numpy array of shape (len(texts), embedding_dim), normalized to unit length.
    """
    model = _model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # cosine sim = dot product after normalization
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string. Returns (D,) float32 array."""
    return embed_texts([query], model_name=model_name)[0]
