#!/usr/bin/env python3
"""Baseline arm of E1: this repo's own retriever, scored by score.py.

TWO JOBS, and the second one is the important one:

  1. Produce the hand-rolled row of the comparison table.
  2. **Validate score.py.** If this run does not reproduce eval_recall.py's
     published hybrid numbers (recall@5 97.94%, MRR 0.943) then score.py is
     wrong, and every framework number it produces is wrong in the same
     direction. Any framework result read before this check passes is
     uninterpretable.

Runs in the MAIN venv (it needs chromadb), unlike the framework arms.

Usage:
    .venv/bin/python evals/frameworks/run_handrolled.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from score import load_cases, report, score          # noqa: E402

# Pinned to whatever `scripts/eval_recall.py` currently reports for hybrid.
# Updated 2026-08-03 from 0.9691/0.936 when the BM25 tokenizer gained stemming
# — that change moved the retriever, so the pin moved with it. It is a
# CONSISTENCY check between two harnesses, not a regression bar; when the
# retriever legitimately changes, re-run eval_recall.py and update both.
PUBLISHED = {"recall": 0.9794, "mrr": 0.943}
OUT = Path(__file__).parent / "results" / "handrolled.json"


def main() -> None:
    from arxiv_rag.config import Config
    from arxiv_rag.index import PaperIndex
    from arxiv_rag.retrieve import retrieve

    cfg = Config()
    index = PaperIndex(cfg)

    # Warm the embedding model and the exact-search matrix BEFORE timing, so
    # the per-query latency measures retrieval and not a one-off model load.
    t_build = time.perf_counter()
    from arxiv_rag.embed import embed_query
    warm = embed_query("warmup", cfg.embed_model, device=cfg.embed_device)
    index.dense_search(warm.tolist(), k=1)
    build_s = time.perf_counter() - t_build

    pos, _ = load_cases()
    latencies: list[float] = []

    def run(query: str) -> list[dict]:
        t0 = time.perf_counter()
        hits = retrieve(query, index, cfg)
        latencies.append((time.perf_counter() - t0) * 1000)
        return hits

    res = score(pos, run)
    latencies.sort()
    res["build_s"] = round(build_s, 2)
    res["p50_ms"] = round(latencies[len(latencies) // 2], 2)
    res["p95_ms"] = round(latencies[int(len(latencies) * 0.95)], 2)

    report("hand-rolled (arxiv_rag.retrieve)", res, {
        "index warm (s)": res["build_s"],
        "query p50 (ms)": res["p50_ms"],
        "query p95 (ms)": res["p95_ms"],
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    # The gate on score.py itself.
    dr = abs(res["recall"] - PUBLISHED["recall"])
    dm = abs(res["mrr"] - PUBLISHED["mrr"])
    if dr > 0.001 or dm > 0.001:
        sys.exit(
            f"\nSCORER MISMATCH — this harness gives "
            f"{res['recall']:.4f}/{res['mrr']:.4f} but eval_recall.py publishes "
            f"{PUBLISHED['recall']}/{PUBLISHED['mrr']}. score.py does not "
            f"reproduce the repo metric, so framework numbers from it are not "
            f"comparable to anything. Fix score.py before reading them."
        )
    print(f"\nscorer validated against eval_recall.py "
          f"({res['recall']:.4f} / {res['mrr']:.4f}) — framework arms are comparable.")


if __name__ == "__main__":
    main()
