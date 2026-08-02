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

    mode = getattr(cfg, "query_expansion", None)
    if mode:
        return retrieve_expanded(query, index, cfg, mode=mode,
                                 n=1 if mode == "hyde" else 3)

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

    # Gate signal, stamped so generate()/api.py can read it without an index.
    top_dense = dense_results[0]["score"] if dense_results else 0.0
    for entry in results:
        entry["relevance"] = float(top_dense)

    if getattr(cfg, "rerank", False):
        from .rerank import rerank as _rerank
        results = _rerank(query, results, top_n=cfg.rerank_top_n,
                          final_k=cfg.final_k, model_name=cfg.rerank_model)

    return results


def retrieve_expanded(query: str, index: PaperIndex, config: Config | None = None,
                      mode: str = "multi", n: int = 3) -> list[dict]:
    """Retrieve over several query phrasings and fuse across all of them.

    RRF extends to this for free: instead of fusing 2 ranked lists (dense, bm25)
    we fuse 2*V lists for V query variants. A chunk that several variants agree
    on accumulates several contributions, which is exactly the consensus
    behaviour RRF already provides between retrievers.

    IMPORTANT — the abstain gate must still be judged on the ORIGINAL query.
    HyDE passages are answer-shaped, so they sit much closer to chunk embeddings
    than a question does; letting them set `score` would inflate similarity for
    every query including off-topic ones and silently break the relevance gate.
    So dense hits keep the ORIGINAL query's cosine in `score`, and variant
    similarities are used only for ranking.
    """
    from .expand import expand_query

    cfg = config or Config()
    rrf_k = getattr(cfg, "rrf_k", _RRF_K)
    variants = expand_query(query, cfg, mode=mode, n=n)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    dense_rank: dict[str, int] = {}
    bm25_rank: dict[str, int] = {}
    origin: dict[str, set] = {}

    for vi, variant in enumerate(variants):
        d = index.dense_search(
            embed_query(variant, cfg.embed_model,
                        device=getattr(cfg, "embed_device", None)).tolist(),
            k=cfg.top_k,
        )
        b = index.bm25_search(variant, k=cfg.top_k)
        for lst, ranks in ((d, dense_rank), (b, bm25_rank)):
            for rank, result in enumerate(lst):
                cid = result["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
                origin.setdefault(cid, set()).add(vi)
                # Only the original query (vi == 0) may define the stored dict,
                # so `score` stays comparable with the un-expanded path.
                if vi == 0:
                    chunk_map[cid] = result
                    ranks.setdefault(cid, rank + 1)
                else:
                    chunk_map.setdefault(cid, {**result, "score": None})

    ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    results = []
    for cid, score in ranked[: cfg.final_k]:
        entry = dict(chunk_map[cid])
        entry["rrf_score"] = round(score, 5)
        entry["dense_rank"] = dense_rank.get(cid)
        entry["bm25_rank"] = bm25_rank.get(cid)
        entry["n_variants"] = len(origin.get(cid, ()))
        results.append(entry)

    # ORIGINAL-query relevance, not the expanded list's. Reading it off the
    # result list drops abstain AUC 0.975 -> 0.826, because RRF across variants
    # frequently evicts the chunk holding the best original-query cosine.
    rel = relevance_score(query, index, cfg)
    for entry in results:
        entry["relevance"] = rel
    return results


# Ablation dispatch. Keyed so scripts/eval_recall.py --mode can select a
# retriever without duplicating the retrieval logic in the harness.
RETRIEVERS = {
    "hybrid": retrieve,
    "dense": retrieve_dense,
    "bm25": retrieve_bm25,
    "expand": lambda q, i, c=None: retrieve_expanded(q, i, c, mode="multi"),
    "hyde": lambda q, i, c=None: retrieve_expanded(q, i, c, mode="hyde", n=1),
}


def relevance_score(query: str, index: PaperIndex,
                    config: Config | None = None) -> float:
    """Abstain signal for `query`: best dense cosine over the WHOLE index.

    Computed from the original query directly rather than read off a result
    list, because the result list is not a stable sample of it. Under query
    expansion the returned top-k is chosen by RRF across all variants, so the
    chunk carrying the best original-query cosine is frequently NOT in it —
    which silently degraded the gate (abstain AUC 0.975 -> 0.828) until this
    was separated out. See NOTES-changes.md §15.
    """
    cfg = config or Config()
    hits = index.dense_search(
        embed_query(query, cfg.embed_model,
                    device=getattr(cfg, "embed_device", None)).tolist(),
        k=1,
    )
    return float(hits[0]["score"]) if hits else 0.0


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
    # Prefer the explicitly stamped signal when present — under query expansion
    # the result list is not a valid sample of original-query similarity.
    stamped = [h["relevance"] for h in hits if isinstance(h.get("relevance"), float)]
    if stamped:
        return max(stamped)
    dense = [h["score"] for h in hits
             if h.get("dense_rank") and isinstance(h.get("score"), (int, float))]
    # Entries contributed only by expansion variants carry score=None by design
    # (see retrieve_expanded) and are correctly excluded by the isinstance check.
    return max(dense) if dense else 0.0
