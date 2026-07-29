"""Hybrid retrieval: fuse BM25 (sparse) + ChromaDB (dense) scores.

Reciprocal Rank Fusion (RRF) is used instead of weighted score normalization
because it's robust to score scale differences between BM25 and cosine similarity.
RRF formula: score(d) = sum(1 / (k + rank(d))) for each retriever.

Why hybrid?
  - BM25 wins on exact technical terms: "GRPO", "LoRA", "HDBSCAN", "vLLM"
  - Dense wins on semantic similarity: "model struggles with multi-property reasoning"
  - Together they cover both lookup-style and exploratory queries.
"""

from __future__ import annotations

from .config import Config
from .embed import embed_query
from .index import PaperIndex


_RRF_K = 60   # standard RRF constant; higher = flatter ranking


def retrieve(query: str, index: PaperIndex, config: Config | None = None) -> list[dict]:
    """Run hybrid retrieval and return top-k chunks ranked by RRF score.

    Args:
        query: Natural-language query string.
        index: Initialized PaperIndex.
        config: Config (uses defaults if None).

    Returns:
        List of chunk dicts sorted by relevance, with 'rrf_score' added.
    """
    cfg = config or Config()

    # 1. Dense retrieval
    q_emb = embed_query(query, cfg.embed_model).tolist()
    dense_results = index.dense_search(q_emb, k=cfg.top_k)

    # 2. BM25 retrieval
    bm25_results = index.bm25_search(query, k=cfg.top_k)

    # 3. RRF fusion
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[cid] = result

    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = result

    # 4. Sort by RRF score, take final_k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ranked[: cfg.final_k]:
        entry = dict(chunk_map[cid])
        entry["rrf_score"] = round(score, 4)
        results.append(entry)

    return results
