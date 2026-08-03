#!/usr/bin/env python3
"""Export the live corpus so every framework retrieves over IDENTICAL chunks.

THIS SCRIPT IS THE FAIRNESS CONTROL, and it is the reason the ablation means
anything.

LlamaIndex and LangChain both ship their own text splitters, and if each one
re-chunked the PDFs, the three implementations would differ in chunk
boundaries, chunk count, and what a "hit" even is. The measured difference
would then be *chunking*, not the abstraction layer — and chunking is not what
E1 claims to compare.

So all three read the same 3288 chunks, with the same chunk_ids, produced by
this repo's font-metric section parser. What is left varying is exactly the
thing under test: how each framework indexes, retrieves, and fuses.

Held constant across all three implementations:
  - chunk text and boundaries      (this file)
  - embedding model                all-MiniLM-L6-v2
  - top_k per retriever            8
  - final_k                        5
  - RRF k                          60
  - eval cases + scoring code      evals/retrieval_cases.json, score.py

Deliberately NOT held constant (these ARE the framework's choices, and
normalising them away would hide the finding):
  - BM25 tokenisation
  - vector-store backend and its search algorithm
  - how the framework's own fusion implementation breaks ties

Usage:
    .venv/bin/python evals/frameworks/export_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arxiv_rag.config import Config
from arxiv_rag.index import PaperIndex

OUT = Path(__file__).parent / "corpus.jsonl"


def main() -> None:
    index = PaperIndex(Config())
    n = index.count()
    if not n:
        sys.exit("index is empty — run scripts/ingest.py first")

    got = index._col.get(include=["documents", "metadatas"])

    # Sort by chunk_id so the export is byte-stable across runs. Chroma does
    # not promise an order, and an unstable corpus file would make BM25 index
    # construction order-dependent between frameworks — a difference that would
    # look like a framework effect.
    rows = sorted(
        zip(got["ids"], got["documents"], got["metadatas"]),
        key=lambda r: r[0],
    )

    with OUT.open("w") as fh:
        for cid, doc, meta in rows:
            fh.write(json.dumps({
                "chunk_id": cid,
                "text": doc,
                "arxiv_id": str(meta.get("arxiv_id", "")),
                "title": str(meta.get("title", "")),
                "section": str(meta.get("section", "")),
                "published": str(meta.get("published", "")),
            }) + "\n")

    papers = len({json.loads(l)["arxiv_id"] for l in OUT.read_text().splitlines()})
    print(f"wrote {OUT}  ·  {len(rows)} chunks · {papers} papers")


if __name__ == "__main__":
    main()
