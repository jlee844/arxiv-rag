#!/usr/bin/env python3
"""LangChain arm of E1: EnsembleRetriever (RRF) over FAISS + BM25.

Runs in `.venv-frameworks`, NOT the app venv.

    .venv-frameworks/bin/python evals/frameworks/run_langchain.py

FRICTION LOG:

  1. **The pieces are spread across a sunset package and a "classic" one.**
     `BM25Retriever` and `FAISS` come from `langchain-community`, which prints
     on import: "langchain-community is being sunset and is no longer actively
     maintained."
  1b. And `EnsembleRetriever` is not in `langchain.retrievers` at all — that
     module was removed in LangChain 1.x. It now lives in
     `langchain_classic.retrievers.ensemble`. So building textbook hybrid
     retrieval on LangChain 1.3 means importing from one *sunset* package and
     one package named *classic*, with no deprecation shim pointing the way.
     Measured on 2026-08-03 against langchain 1.3.14 / community 0.4.2; this is
     ecosystem churn, not a permanent property.
  2. `EnsembleRetriever` does RRF with `c=60`, matching this repo's `rrf_k=60`
     — but `c` is a constructor arg here, unlike LlamaIndex where it is
     hardcoded. Point to LangChain.
  3. **BM25Retriever has no `k` at fusion time in the way you expect.** Each
     child retriever's own `k` governs its candidate list; the ensemble then
     fuses and truncates. Getting "top 8 per retriever, fuse, keep 5" requires
     setting `k` on each child AND slicing the ensemble output, because the
     ensemble returns the full fused union rather than a top-n.
  4. **Default BM25 tokenisation is `text.split()`** — the same naive scheme
     this repo uses, and notably NOT the stemmed/stopworded tokeniser
     LlamaIndex defaults to. This is the single biggest driver of the score
     difference between the two frameworks, and it is a default nobody would
     think to compare.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from score import load_cases, report, score          # noqa: E402

HERE = Path(__file__).parent
CORPUS = HERE / "corpus.jsonl"
OUT = HERE / "results" / "langchain.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8
FINAL_K = 5
RRF_C = 60       # matches Config.rrf_k


def main() -> None:
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    # friction #1b: NOT `langchain.retrievers` — LangChain 1.x removed that
    # module outright. `EnsembleRetriever` now lives in `langchain_classic`.
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    print(f"{len(rows)} chunks from {CORPUS.name}")

    docs = [
        Document(
            page_content=r["text"],
            metadata={"arxiv_id": r["arxiv_id"], "title": r["title"],
                      "section": r["section"], "chunk_id": r["chunk_id"]},
        )
        for r in rows
    ]

    t0 = time.perf_counter()
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # LangChain embeds `page_content` only — metadata is not concatenated in,
    # so unlike LlamaIndex there is no contamination default to defuse here.
    store = FAISS.from_documents(docs, embed)
    dense = store.as_retriever(search_kwargs={"k": TOP_K})

    sparse = BM25Retriever.from_documents(docs)
    sparse.k = TOP_K                       # friction #3: set on the child

    ensemble = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.5, 0.5],
        c=RRF_C,
    )
    build_s = time.perf_counter() - t0
    print(f"built in {build_s:.1f}s")

    pos, _ = load_cases()
    latencies: list[float] = []

    def run(query: str) -> list[dict]:
        t = time.perf_counter()
        hits = ensemble.invoke(query)
        latencies.append((time.perf_counter() - t) * 1000)
        # friction #3: the ensemble returns the whole fused union, so the
        # final_k truncation is the caller's job.
        return [{"arxiv_id": d.metadata["arxiv_id"],
                 "chunk_id": d.metadata["chunk_id"]} for d in hits[:FINAL_K]]

    res = score(pos, run, k=FINAL_K)
    latencies.sort()
    res["build_s"] = round(build_s, 2)
    res["p50_ms"] = round(latencies[len(latencies) // 2], 2)
    res["p95_ms"] = round(latencies[int(len(latencies) * 0.95)], 2)

    report("LangChain (EnsembleRetriever, FAISS + BM25)", res, {
        "index build (s)": res["build_s"],
        "query p50 (ms)": res["p50_ms"],
        "query p95 (ms)": res["p95_ms"],
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
