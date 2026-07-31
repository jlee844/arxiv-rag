#!/usr/bin/env python3
"""Latency / throughput bench for the retrieval stack.

Breaks out: embed | dense | bm25 | fuse+total
Reports p50 / p95 over repeated queries. No LLM (generation is separate).

Usage:
    .venv/bin/python scripts/bench_latency.py
    .venv/bin/python scripts/bench_latency.py --rounds 20
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from arxiv_rag.config import Config
from arxiv_rag.embed import embed_query
from arxiv_rag.index import PaperIndex
from arxiv_rag.retrieve import retrieve

QUERIES = [
    "what benchmarks evaluate hallucination in VLMs?",
    "POPE object hallucination",
    "remote sensing captioning satellite",
    "causal tracing BLIP",
    "what is the capital of France?",
]


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round((p / 100) * (len(xs) - 1)))))
    return xs[i]


@click.command()
@click.option("--rounds", default=10, show_default=True)
def main(rounds: int) -> None:
    cfg = Config()
    index = PaperIndex(cfg)
    n = index.count()
    click.echo(f"index_chunks={n}  papers≈{len(index.indexed_papers())}  rounds={rounds}")

    # warmup (MPS / model load)
    retrieve(QUERIES[0], index, cfg)

    embed_ms, dense_ms, bm25_ms, total_ms = [], [], [], []

    for r in range(rounds):
        for q in QUERIES:
            t0 = time.perf_counter()
            vec = embed_query(q, cfg.embed_model).tolist()
            t1 = time.perf_counter()
            index.dense_search(vec, k=cfg.top_k)
            t2 = time.perf_counter()
            index.bm25_search(q, k=cfg.top_k)
            t3 = time.perf_counter()
            retrieve(q, index, cfg)
            t4 = time.perf_counter()

            embed_ms.append((t1 - t0) * 1000)
            dense_ms.append((t2 - t1) * 1000)
            bm25_ms.append((t3 - t2) * 1000)
            total_ms.append((t4 - t3) * 1000)  # full retrieve path

    def row(name: str, xs: list[float]) -> None:
        click.echo(
            f"{name:10s}  p50={_pct(xs,50):7.1f}ms  p95={_pct(xs,95):7.1f}ms  "
            f"mean={statistics.mean(xs):7.1f}ms  n={len(xs)}"
        )

    click.echo("─" * 60)
    row("embed", embed_ms)
    row("dense", dense_ms)
    row("bm25", bm25_ms)
    row("retrieve", total_ms)
    qps = 1000.0 / statistics.mean(total_ms) if total_ms else 0
    click.echo(f"approx QPS (retrieve only, serial): {qps:.1f}")
    click.echo(
        "scalability note: BM25 rebuild is O(N) at ingest; "
        f"search is O(N) over {n} chunks — re-bench after 10x ingest."
    )


if __name__ == "__main__":
    main()