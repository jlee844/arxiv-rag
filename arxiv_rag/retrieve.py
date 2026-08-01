"""Hybrid retrieval: fuse BM25 (sparse) + ChromaDB (dense) rankings.

Reciprocal Rank Fusion is used instead of weighted score normalization because
BM25 and cosine similarity live on incompatible scales:

  - BM25 returns unbounded positive scores (0 to ~30 here), and they are
    CORPUS-RELATIVE — they shift when documents are added or removed.
  - Cosine similarity is bounded roughly 0..1.

Adding them lets BM25 dominate purely because its numbers are bigger, which is
a scale artifact, not a relevance signal. RRF throws the scores away and fuses
RANKS instead, so no calibration and no per-corpus weight tuning is needed:

    RRF(d) = sum over retrievers i of  1 / (k + rank_i(d))

Why hybrid at all?
  - BM25 wins on exact technical tokens: "GRPO", "LoRA", "HDBSCAN", "vLLM".
    Rare tokens are smeared by the embedder but are exactly where BM25 is
    strongest.
  - Dense wins on paraphrase: "model struggles with multi-property reasoning".
  - They fail in opposite directions, which is the whole argument for running
    both.
"""

from __future__ import annotations

from .config import Config
from .embed import embed_query
from .index import PaperIndex


_RRF_K = 60   # default when no Config is supplied; see Config.rrf_k


def retrieve_dense(query: str, index: PaperIndex,
                   config: Config | None = None) -> list[dict]:
    """Dense-only retrieval. Ablation baseline — NOT the production path.

    Fetches top_k then truncates to final_k so it competes under exactly the
    same answer budget as hybrid. Anything else would make the comparison
    meaningless.
    """
    cfg = config or Config()
    hits = index.dense_search(
        embed_query(query, cfg.embed_model,
                    device=getattr(cfg, "embed_device", None)).tolist(),
        k=cfg.top_k
    )
    for rank, h in enumerate(hits, 1):
        h["dense_rank"], h["bm25_rank"] = rank, None
    return hits[: cfg.final_k]


def retrieve_bm25(query: str, index: PaperIndex,
                  config: Config | None = None) -> list[dict]:
    """BM25-only retrieval. Ablation baseline — NOT the production path."""
    cfg = config or Config()
    hits = index.bm25_search(query, k=cfg.top_k)
    for rank, h in enumerate(hits, 1):
        h["dense_rank"], h["bm25_rank"] = None, rank
    return hits[: cfg.final_k]


def retrieve(query: str, index: PaperIndex, config: Config | None = None) -> list[dict]:
    """Run hybrid retrieval and return top chunks ranked by RRF score.

    Args:
        query: Natural-language query string.
        index: Initialized PaperIndex.
        config: Config (uses defaults if None).

    Returns:
        List of chunk dicts sorted by relevance, each with 'rrf_score' plus
        'dense_rank' / 'bm25_rank' (1-based, or None if that retriever missed
        it) so you can see WHICH retriever found each result.
    """
    cfg = config or Config()
    rrf_k = getattr(cfg, "rrf_k", _RRF_K)

    dense_results = index.dense_search(
        embed_query(query, cfg.embed_model,
                    device=getattr(cfg, "embed_device", None)).tolist(),
        k=cfg.top_k
    )
    bm25_results = index.bm25_search(query, k=cfg.top_k)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    dense_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_map[cid] = result
        dense_rank[cid] = rank + 1

    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_map.setdefault(cid, result)
        bm25_rank[cid] = rank + 1

    ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)

    # When reranking, keep a wider candidate list for the cross-encoder to
    # rescore; otherwise it can only reorder what fusion already chose.
    keep = getattr(cfg, "rerank_top_n", 20) if getattr(cfg, "rerank", False) else cfg.final_k

    results = []
    for cid, score in ranked[:keep]:
        entry = dict(chunk_map[cid])
        entry["rrf_score"] = round(score, 5)
        entry["dense_rank"] = dense_rank.get(cid)
        entry["bm25_rank"] = bm25_rank.get(cid)
        results.append(entry)

    if getattr(cfg, "rerank", False):
        from .rerank import rerank as _rerank
        results = _rerank(query, results, top_n=cfg.rerank_top_n,
                          final_k=cfg.final_k, model_name=cfg.rerank_model)

    return results


# Ablation dispatch. Keyed so scripts/eval_recall.py --mode can select a
# retriever without duplicating the retrieval logic in the harness.
RETRIEVERS = {
    "hybrid": retrieve,
    "dense": retrieve_dense,
    "bm25": retrieve_bm25,
}


def max_dense_score(hits: list[dict]) -> float:
    """Best dense cosine similarity among hits, or 0.0 if dense found none.

    Only entries carrying a dense_rank have a cosine `score`; BM25-only entries
    carry an unbounded BM25 score, and mixing the two would let a BM25 score of
    ~20 sail past any cosine threshold.

    Approximation worth naming: this reads the post-fusion top-`final_k`, not
    the full dense candidate set. Dense rank 1 could in principle be pushed out
    of the final list by 5 consensus chunks — but a query with 5 chunks both
    retrievers agree on is unambiguously on-topic, so the gate would pass anyway.
    """
    dense = [h["score"] for h in hits
             if h.get("dense_rank") and isinstance(h.get("score"), (int, float))]
    return max(dense) if dense else 0.0
