"""Query expansion — rewrite the query before retrieval.

MOTIVATION (NOTES-changes.md §14): `paraphrase-hallucination` is missed by every
retriever, every top_k, the cross-encoder, and a larger embedder. Root cause is
not semantic — the target chunk ranks 11 of 2960 in the full dense ordering — but
structural: RRF at K=60 is flat enough that a deep single-retriever hit can never
climb into the top 5.

So the fix has to change *retrieval*, not fusion. Two ways to do that:

  "multi"  Generate alternative phrasings, retrieve for each, fuse across all.
           Aimed at BM25: the query says "invent objects that are not in the
           image" while the paper says "hallucination", so there is zero lexical
           overlap and BM25 contributes nothing. A rewrite that guesses the
           field's actual terminology gives BM25 something to match.

  "hyde"   Hypothetical Document Embeddings. Ask the LLM to write the passage
           that WOULD answer the question, then embed that instead of the
           question. Aimed at dense: questions and answer passages occupy
           different regions of embedding space, so a question embedding is a
           biased probe for answer-shaped text.

COST: both add an LLM call BEFORE retrieval. Retrieval is ~8 ms; an Ollama call
is ~1-2 s. That is a 100-250x increase on the retrieval path, and it is only
defensible because generation already costs seconds. It would NOT be defensible
on /api/search, which exists precisely to be fast.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .config import Config

_MULTI_PROMPT = """\
You rewrite search queries for a corpus of machine-learning research papers.

Given a user question, write {n} alternative phrasings that a paper answering it \
would be likely to use. Prefer the field's standard technical terminology, \
including the specific term of art the question is avoiding.

Rules:
- Each rewrite is a standalone search query, 4 to 15 words.
- Do NOT answer the question.
- Do NOT number them or add commentary.
- One rewrite per line, nothing else.

Question: {query}
"""

_HYDE_PROMPT = """\
Write a short passage (2-3 sentences) that would plausibly appear in a machine \
learning research paper and would answer the question below.

Write it as an excerpt from a paper — declarative, technical, using standard \
terminology. Invented specifics are fine; this text is used only as a retrieval \
probe and is never shown to anyone.

Do not preface it. Output the passage only.

Question: {query}
"""


def _chat(prompt: str, cfg: Config) -> str:
    import ollama
    client = ollama.Client(host=cfg.ollama_base_url)
    resp = client.chat(
        model=cfg.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        # Low but non-zero: rewrites should vary from each other, not from run
        # to run. 0.0 tends to produce n near-identical restatements.
        options={"temperature": 0.3},
    )
    return resp["message"]["content"].strip()


def _clean_lines(text: str, limit: int) -> list[str]:
    """Strip numbering/bullets/quotes the model adds despite being told not to."""
    out = []
    for line in text.splitlines():
        line = re.sub(r'^\s*(?:\d+[.)]|[-*•])\s*', "", line).strip().strip('"\'')
        if 3 < len(line.split()) <= 25:
            out.append(line)
    return out[:limit]


@lru_cache(maxsize=512)
def _expand_cached(query: str, mode: str, n: int, model: str, host: str) -> tuple[str, ...]:
    """Cached because expansion is deterministic-ish and costs a full LLM call.

    Keyed on model+host too, so switching backends doesn't serve stale rewrites.
    """
    cfg = Config()
    cfg.ollama_model, cfg.ollama_base_url = model, host
    if mode == "hyde":
        passage = _chat(_HYDE_PROMPT.format(query=query), cfg)
        return (passage,) if passage else ()
    raw = _chat(_MULTI_PROMPT.format(query=query, n=n), cfg)
    return tuple(_clean_lines(raw, n))


def expand_query(query: str, config: Config | None = None,
                 mode: str = "multi", n: int = 3) -> list[str]:
    """Return [original, *expansions]. The original is ALWAYS first and kept.

    Keeping the original is not optional: expansion can drift off-topic, and
    without the original as an anchor a bad rewrite can lose a query that plain
    retrieval would have answered.
    """
    cfg = config or Config()
    try:
        extra = _expand_cached(query, mode, n, cfg.ollama_model, cfg.ollama_base_url)
    except Exception:
        # Expansion is an enhancement; if the LLM is unreachable, degrade to
        # plain retrieval rather than failing the request.
        return [query]
    return [query, *(e for e in extra if e.lower() != query.lower())]
