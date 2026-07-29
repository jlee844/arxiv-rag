"""Embedding model wrapper — CPU-only, cached singleton.

STAGE 3 of PLAN-learn.md — the easiest stage, ~30 lines.
Reference: `git show 4c7f66a:arxiv_rag/embed.py`

Model: all-MiniLM-L6-v2
  - 22 MB download, 384-dim embeddings
  - ~1000 sentences/sec on modern CPU, no GPU
  - Upgrade path: "all-mpnet-base-v2" (768-dim, ~5% better recall, ~2x slower)

Don't chase embedding quality before you can measure it (see Stage 8 extensions).
"""

from __future__ import annotations
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model(model_name: str):
    """Load the SentenceTransformer once and cache it.

    WHY the cache: constructing SentenceTransformer reads ~90 MB from disk and
    takes 2-3s. embed_texts() is called once per batch — dozens of times per
    ingest. Without @lru_cache you reload the model every call and ingest goes
    from seconds to minutes. This one decorator is the whole difference.

    WHY the import is inside the function: sentence_transformers pulls in torch
    (~2s). Modules that only need Config or Chunk shouldn't pay that cost.

    TODO: import SentenceTransformer here, return SentenceTransformer(model_name).
    """
    raise NotImplementedError


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed a list of strings. Returns (N, D) float32 array, L2-normalized.

    TODO: call _model(model_name).encode(...) with
      batch_size=batch_size
      show_progress_bar=show_progress
      normalize_embeddings=True   <- see below
      convert_to_numpy=True
    then .astype(np.float32).

    WHY normalize_embeddings=True: after L2 normalization, cosine similarity IS
    the dot product — the division drops out. Chroma's hnsw:space=cosine then
    works correctly and faster. Nearly free, so always on.
    """
    raise NotImplementedError


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string. Returns (D,) float32 array.

    TODO: one line in terms of embed_texts. Mind the shape — callers want (D,),
    not (1, D).
    """
    raise NotImplementedError
