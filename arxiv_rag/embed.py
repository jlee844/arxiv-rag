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


@lru_cache(maximize=1)
def _best_device() -> str:
  """Pick fastest available torch backend.

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
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
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


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string. Returns (D,) float32 array.

    TODO: one line in terms of embed_texts. Mind the shape — callers want (D,),
    not (1, D).
    """
    return embed_texts([query], model_name=model_name)[0]
