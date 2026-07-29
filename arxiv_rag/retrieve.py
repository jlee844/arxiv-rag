"""Hybrid retrieval: fuse BM25 (sparse) + ChromaDB (dense) results.

STAGE 5 of PLAN-learn.md — the intellectual core. ~40 lines, the most important
40 in the project. Reference: `git show 4c7f66a:arxiv_rag/retrieve.py`

THE PROBLEM
  BM25 returns unbounded positive scores (0 to ~30, corpus-dependent).
  Cosine similarity returns roughly 0 to 1.
  You CANNOT add these. Try it: BM25 dominates purely because its numbers are
  bigger. That's a scale artifact, not a relevance signal.

TWO WAYS OUT
  a) Normalize then weight: min-max each list to [0,1], then
     0.65*dense + 0.35*sparse. Needs per-corpus tuning, and min-max is unstable
     when one list is nearly uniform.
  b) Reciprocal Rank Fusion: throw the SCORES away, use only the RANKS.

        RRF(d) = sum over retrievers i of  1 / (k + rank_i(d))

  RRF wins because scale-invariance is free — ranks are ranks. No calibration,
  no tuning. A doc ranked #1 by both retrievers scores 1/61 + 1/61 = 0.0328;
  ranked #1 by one and absent from the other, 0.0164. Agreement across
  retrievers is rewarded automatically, which is exactly what you want and
  never had to hand-tune.

WHY HYBRID AT ALL
  BM25 wins on exact technical terms: "GRPO", "LoRA", "HDBSCAN", "vLLM".
    Dense embeddings compress meaning into 384 floats, and rare tokens get
    smeared into near-neighbours during that compression.
  Dense wins on semantics: "model struggles with multi-property reasoning".
  They fail in opposite directions. That's the whole argument.
"""

from __future__ import annotations

from .config import Config
from .embed import embed_query
from .index import PaperIndex


# Standard RRF constant from the original paper.
#
# What it does: flattens the curve. Without it rank 1 (1/1) would be worth twice
# rank 2 (1/2) — far too peaked, letting one retriever's top hit dominate. At
# k=60 ranks 1 and 2 differ by under 2%, so CONSENSUS matters more than any one
# retriever's confidence.
#
# Exercise: try 1 and 1000, watch the rankings change.
_RRF_K = 60


def retrieve(query: str, index: PaperIndex, config: Config | None = None) -> list[dict]:
    """Run hybrid retrieval and return top-k chunks ranked by RRF score.

    Args:
        query: Natural-language query string.
        index: Initialized PaperIndex.
        config: Config (uses defaults if None).

    Returns:
        List of chunk dicts sorted by relevance, each with 'rrf_score' added.

    TODO:
      1. Dense: embed_query(query, cfg.embed_model).tolist() -> index.dense_search(k=cfg.top_k)
      2. Sparse: index.bm25_search(query, k=cfg.top_k)
      3. Fuse. Walk each result list with enumerate() for the rank, and
         accumulate into a {chunk_id: score} dict:
             scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
         The `+ 1` matters — enumerate is 0-based, ranks are 1-based.
         Keep a parallel {chunk_id: chunk_dict} so you can rebuild the payload.
      4. Sort by score descending, take cfg.final_k, attach 'rrf_score'.

    Note you retrieve cfg.top_k (8) from EACH retriever but return only
    cfg.final_k (5). Fusion needs a deeper candidate pool than it emits —
    that headroom is where the consensus signal comes from.

    CHECKPOINT: find a rare technical token in your corpus and query it.
    Compare index.dense_search alone vs retrieve(). If hybrid doesn't win on
    rare tokens, your BM25 path is broken — check tokenization first.
    """
    raise NotImplementedError
