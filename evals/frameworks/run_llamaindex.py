#!/usr/bin/env python3
"""LlamaIndex arm of E1: hybrid BM25 + dense with RRF fusion.

Runs in `.venv-frameworks`, NOT the app venv — LlamaIndex pulls a large and
fast-moving dependency tree, and the whole point of isolating it is that a
framework version bump cannot break the shipped retriever.

    .venv-frameworks/bin/python evals/frameworks/run_llamaindex.py

FRICTION LOG — things the abstraction made harder, recorded as they were hit,
because "what breaks when you need something non-standard" is half of what E1
is measuring:

  1. `QueryFusionRetriever` defaults to `num_queries=4`, which calls an **LLM**
     to generate query variants before fusing. That is query expansion, not
     retriever fusion — a different technique, which this repo separately
     measured and rejected. Left at the default it would have (a) required an
     LLM for a pure-retrieval benchmark, (b) made the comparison against
     hand-rolled RRF invalid, and (c) still produced a plausible number. Pinned
     to `num_queries=1`.
  1b. ...and pinning it is not enough. `__init__` resolves `Settings.llm`
     EAGERLY, before it knows whether any generation will happen, so
     constructing the retriever raised `ImportError: llama-index-llms-openai
     package not found` at num_queries=1. A pure-retrieval benchmark cannot
     instantiate the fusion retriever without either installing an LLM
     integration or injecting `MockLLM()`. The dependency is structural, not
     behavioural — the LLM is never called, it just has to exist.
  2. `mode=RECIPROCAL_RANK` uses a hardcoded RRF constant k=60 that is not a
     constructor argument. It happens to match this repo's `rrf_k=60`, so the
     comparison survives — but by luck, not by control.
  3. Nodes must carry `arxiv_id` in metadata AND that metadata must be excluded
     from the embedded text, or the embedding sees the id string and the
     retrieval is contaminated. `excluded_embed_metadata_keys` does this and
     defaults to empty, i.e. contamination is the default.
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
# Variant-specific, so the stemmed and no-stem runs cannot overwrite each
# other — they are the two halves of the arm's own ablation.
def out_path(no_stem: bool):
    return HERE / "results" / ("llamaindex-nostem.json" if no_stem else "llamaindex.json")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8        # per retriever, matching Config.top_k
FINAL_K = 5      # matching Config.final_k


class CountingMockLLM:
    """Wraps MockLLM and counts calls, so "the LLM is never used" is ASSERTED
    rather than assumed.

    If `num_queries=1` did not actually disable generation, a MockLLM would
    happily return empty strings, fusion would run over junk variants, and the
    result would be a believable-but-meaningless number — the exact failure
    shape this project keeps finding. Counting turns that into an exception.
    """

    def __new__(cls, base):
        # MockLLM is a pydantic model; subclassing to add a counter fights the
        # schema. Instead, patch the instance's `complete`/`chat` at runtime.
        cls.calls = 0
        for name in ("complete", "chat", "acomplete", "achat"):
            orig = getattr(base, name, None)
            if orig is None:
                continue

            def wrapper(*a, _orig=orig, **kw):
                CountingMockLLM.calls += 1
                return _orig(*a, **kw)

            object.__setattr__(base, name, wrapper)
        return base


def main() -> None:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.llms import MockLLM
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    print(f"{len(rows)} chunks from {CORPUS.name}")

    nodes = [
        TextNode(
            id_=r["chunk_id"],
            text=r["text"],
            metadata={"arxiv_id": r["arxiv_id"], "title": r["title"],
                      "section": r["section"]},
            # Friction #3: without this the metadata is prepended to the text
            # that gets embedded, so the arxiv id leaks into the vector.
            excluded_embed_metadata_keys=["arxiv_id", "title", "section"],
            excluded_llm_metadata_keys=["arxiv_id"],
        )
        for r in rows
    ]

    t0 = time.perf_counter()
    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    vindex = VectorStoreIndex(nodes, embed_model=embed, show_progress=True)
    dense = vindex.as_retriever(similarity_top_k=TOP_K)
    # ABLATION WITHIN THE ARM: `--no-stem` disables LlamaIndex's default
    # Snowball stemming + stopword removal, leaving whitespace-ish tokens close
    # to this repo's `text.lower().split()`. This is what isolates "the
    # framework retrieves better" from "the framework's BM25 tokeniser is
    # better", which are very different conclusions with very different fixes.
    no_stem = "--no-stem" in sys.argv
    sparse = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=TOP_K, skip_stemming=no_stem,
    )

    fusion = QueryFusionRetriever(
        [dense, sparse],
        similarity_top_k=FINAL_K,
        # Friction #1: the default of 4 calls an LLM to invent query variants.
        num_queries=1,
        # Friction #1b: required even though num_queries=1 means it is never
        # called. Asserted unused below.
        llm=CountingMockLLM(MockLLM()),
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )
    build_s = time.perf_counter() - t0
    print(f"built in {build_s:.1f}s")

    pos, _ = load_cases()
    latencies: list[float] = []

    def run(query: str) -> list[dict]:
        t = time.perf_counter()
        hits = fusion.retrieve(query)
        latencies.append((time.perf_counter() - t) * 1000)
        return [{"arxiv_id": h.node.metadata["arxiv_id"],
                 "chunk_id": h.node.id_, "score": h.score} for h in hits]

    res = score(pos, run, k=FINAL_K)

    # The assertion that keeps this arm honest: if the LLM was touched, this
    # measured query expansion, not retriever fusion.
    if CountingMockLLM.calls:
        sys.exit(f"\nINVALID RUN — the MockLLM was called "
                 f"{CountingMockLLM.calls} times, so num_queries=1 did NOT "
                 f"disable query generation. These numbers measure fusion over "
                 f"LLM-generated variants of empty strings, not hybrid "
                 f"retrieval, and are not comparable to the other arms.")
    res["llm_calls"] = CountingMockLLM.calls

    latencies.sort()
    res["build_s"] = round(build_s, 2)
    res["p50_ms"] = round(latencies[len(latencies) // 2], 2)
    res["p95_ms"] = round(latencies[int(len(latencies) * 0.95)], 2)

    label = "LlamaIndex (fusion, BM25 stemming OFF)" if no_stem else \
            "LlamaIndex (fusion, BM25 stemming ON = default)"
    res["stemming"] = not no_stem
    report(label, res, {
        "index build (s)": res["build_s"],
        "query p50 (ms)": res["p50_ms"],
        "query p95 (ms)": res["p95_ms"],
    })

    out = out_path(no_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
