"""Cross-encoder reranking.

WHY a cross-encoder, given we already have dense + BM25:

A bi-encoder (embed.py) encodes query and document SEPARATELY, then compares
vectors. It can only measure topical proximity — whether two texts are about
similar things. A cross-encoder feeds query and document through the model
TOGETHER, so attention runs across both, and it can judge whether the document
actually ANSWERS the query.

That distinction is exactly what the cosine relevance gate could not make.
Measured on 22 adversarial negatives, "what learning rate should I use with the
Adam optimizer" scored 0.6231 cosine — above four genuine positives — because
it is saturated with corpus vocabulary. Topically close, but the corpus does
not answer it. See NOTES-changes.md §10.

COST: cross-encoders do not scale. Scoring is O(candidates) full forward passes,
so it cannot run over the corpus — only over a shortlist that cheap retrieval
already narrowed. Hence rerank_top_n.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _model(model_name: str, device: str):
    """Load once and cache. ~5.5s cold; unusable per-request without this."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=device)


def rerank_scores(query: str, texts: list[str],
                  model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                  device: str = "cpu") -> list[float]:
    """Raw relevance logits for (query, text) pairs. Higher = more relevant.

    Unbounded logits, NOT probabilities — roughly -11 to +11 for this model.
    Deliberately not squashed through a sigmoid: the raw range is wider and
    therefore a better threshold signal, which is the whole point here.

    device defaults to cpu: candidate lists are ~20 items, and at that batch
    size MPS transfer overhead outweighs the compute (same asymmetry measured
    for the bi-encoder in embed.py).
    """
    if not texts:
        return []
    model = _model(model_name, device)
    scores = model.predict([(query, t) for t in texts])
    return [float(s) for s in scores]


def rerank(query: str, hits: list[dict], top_n: int = 20, final_k: int = 5,
           model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
           device: str = "cpu") -> list[dict]:
    """Rescore the top_n candidates and return the best final_k.

    Adds 'rerank_score' to every returned hit and preserves the original
    fusion provenance (rrf_score / dense_rank / bm25_rank) so the ablation can
    still see which retriever surfaced each candidate.
    """
    if not hits:
        return []

    candidates = hits[:top_n]
    scores = rerank_scores(query, [h["text"] for h in candidates],
                           model_name=model_name, device=device)

    scored = []
    for h, s in zip(candidates, scores):
        entry = dict(h)
        entry["rerank_score"] = round(s, 4)
        scored.append(entry)

    # Stable sort: ties keep their fusion order rather than an arbitrary one.
    scored.sort(key=lambda e: e["rerank_score"], reverse=True)
    return scored[:final_k]


def max_rerank_score(hits: list[dict]) -> float:
    """Best cross-encoder score among hits, or -inf if none were scored.

    Companion to retrieve.max_dense_score, for use as an abstain signal.
    """
    scores = [h["rerank_score"] for h in hits if "rerank_score" in h]
    return max(scores) if scores else float("-inf")
