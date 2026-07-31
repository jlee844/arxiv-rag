#!/usr/bin/env python3
"""Retrieval quality eval with hard slices.

Metrics:
  - recall@k  (positive cases)
  - MRR       (mean reciprocal rank of first relevant paper)
  - per-tag breakdown
  - negative: max RRF of top hit (lower = better separation)

Usage:
    .venv/bin/python scripts/eval_recall.py
    .venv/bin/python scripts/eval_recall.py --k 5 --tag paraphrase
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from arxiv_rag.config import Config
from arxiv_rag.index import PaperIndex
from arxiv_rag.retrieve import RETRIEVERS, retrieve

CASES = Path(__file__).parent.parent / "evals" / "retrieval_cases.json"


def _base_id(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def _first_relevant_rank(chunks: list[dict], want: set[str]) -> int | None:
    """1-based rank of first chunk whose paper is in want; None if miss."""
    seen = []
    for i, c in enumerate(chunks, 1):
        bid = _base_id(c["arxiv_id"])
        if bid in want and bid not in seen:
            return i
        seen.append(bid)
    # paper-level: scan unique paper order
    papers = []
    for c in chunks:
        bid = _base_id(c["arxiv_id"])
        if bid not in papers:
            papers.append(bid)
    for i, bid in enumerate(papers, 1):
        if bid in want:
            return i
    return None


def score_run(data, index, cfg, mode: str = "hybrid", verbose: bool = True) -> dict:
    """Score one retriever over all cases. Returns a summary dict.

    Extracted from main() so --ablate can call it three times without the
    harness re-implementing retrieval per mode.
    """
    retriever = RETRIEVERS[mode]

    pos_hits = 0
    pos_n = 0
    rr_sum = 0.0
    by_tag = defaultdict(lambda: {"hits": 0, "n": 0, "rr": 0.0})
    neg_scores = []

    for case in data:
        chunks = retriever(case["query"], index, cfg)
        want = set(case.get("relevant") or [])
        t = case.get("tag", "untagged")
        cid = case.get("id", "?")

        if not want:
            # negative / OOD: report top RRF; no "relevant" to hit
            top = (chunks[0].get("rrf_score", chunks[0].get("score", 0.0))
                   if chunks else 0.0)
            neg_scores.append(top)
            if verbose:
                click.echo(f"[NEG] {cid:28s}  top_score={top:.5f}  tag={t}")
            continue

        pos_n += 1
        rank = _first_relevant_rank(chunks, want)
        ok = rank is not None and rank <= cfg.final_k
        pos_hits += int(ok)
        rr = 1.0 / rank if rank else 0.0
        rr_sum += rr
        by_tag[t]["n"] += 1
        by_tag[t]["hits"] += int(ok)
        by_tag[t]["rr"] += rr

        if verbose:
            got = {_base_id(c["arxiv_id"]) for c in chunks}
            mark = "OK" if ok else "MISS"
            click.echo(
                f"[{mark}] {cid:28s}  rank={rank}  rr={rr:.2f}  "
                f"want={sorted(want)} got={sorted(got)}"
            )

    return {
        "mode": mode,
        "recall": pos_hits / pos_n if pos_n else 0.0,
        "mrr": rr_sum / pos_n if pos_n else 0.0,
        "hits": pos_hits,
        "n": pos_n,
        "by_tag": {t: dict(v) for t, v in by_tag.items()},
        "neg_mean": sum(neg_scores) / len(neg_scores) if neg_scores else None,
        "neg_n": len(neg_scores),
    }


def print_summary(r: dict, k: int) -> None:
    click.echo("─" * 60)
    click.echo(f"[{r['mode']}]  recall@{k}: {r['hits']}/{r['n']} = {r['recall']:.2%}")
    click.echo(f"MRR:              {r['mrr']:.3f}")
    for t, s_ in sorted(r["by_tag"].items()):
        if s_["n"]:
            click.echo(
                f"  tag={t:12s}  recall={s_['hits']/s_['n']:.2%}  "
                f"MRR={s_['rr']/s_['n']:.3f}  n={s_['n']}"
            )
    if r["neg_mean"] is not None:
        click.echo(
            f"negatives: n={r['neg_n']}  mean_top={r['neg_mean']:.5f}  "
            f"(RRF consensus max≈0.0328; single≈0.0164)"
        )


def _tags(runs: list[dict]) -> list[str]:
    return sorted({t for r in runs for t in r["by_tag"]})


def print_ablation(runs: list[dict], k: int) -> None:
    tags = _tags(runs)
    head = f"| {'retriever':10} | {'recall@'+str(k):9} | {'MRR':5} | " + " | ".join(
        f"{t:10}" for t in tags) + " |"
    click.echo("\n" + head)
    click.echo("|" + "-" * (len(head) - 2) + "|")
    for r in runs:
        cells = []
        for t in tags:
            s_ = r["by_tag"].get(t)
            cells.append(f"{s_['hits']/s_['n']:.0%}".rjust(10) if s_ and s_["n"]
                         else " " * 10)
        click.echo(
            f"| {r['mode']:10} | {r['recall']:9.2%} | {r['mrr']:.3f} | "
            + " | ".join(cells) + " |"
        )
    click.echo()


@click.command()
@click.option("--k", default=None, type=int)
@click.option("--cases", default=str(CASES), type=click.Path(exists=True))
@click.option("--tag", default=None, help="Filter to one tag")
@click.option("--mode", default="hybrid",
              type=click.Choice(["hybrid", "dense", "bm25"]),
              help="Which retriever to score")
@click.option("--ablate", is_flag=True,
              help="Score dense / bm25 / hybrid and print a comparison table")
@click.option("--rrf-k", "rrf_k_sweep", default=None,
              help="Comma-separated RRF k values to sweep, e.g. 1,10,60,200")
@click.option("--device", default="cpu",
              help="Embedding device. Default 'cpu' for REPRODUCIBILITY: mps "
                   "differs by ~2e-7/component, enough to flip near-tied HNSW "
                   "neighbours and move recall a full case. Use 'mps' to "
                   "measure the production path instead.")
@click.option("--repeat", default=1, type=int,
              help="Run N times and report min/max — determinism guard.")
def main(k, cases, tag, mode, ablate, rrf_k_sweep, device, repeat) -> None:
    cfg = Config()
    cfg.embed_device = None if device == "auto" else device
    if k is not None:
        cfg.final_k = k
    index = PaperIndex(cfg)
    if index.count() == 0:
        click.echo("Index empty.", err=True)
        sys.exit(1)

    data = json.loads(Path(cases).read_text())
    if tag:
        data = [c for c in data if c.get("tag") == tag]

    click.echo(f"corpus: {index.count()} chunks / {len(index.indexed_papers())} papers")

    if rrf_k_sweep:
        runs = []
        for val in [int(v) for v in rrf_k_sweep.split(",")]:
            cfg.rrf_k = val
            r = score_run(data, index, cfg, "hybrid", verbose=False)
            r["mode"] = f"rrf_k={val}"
            runs.append(r)
        print_ablation(runs, cfg.final_k)
        return

    if ablate:
        trials = [[score_run(data, index, cfg, m, verbose=False)
                   for m in ("dense", "bm25", "hybrid")] for _ in range(repeat)]
        runs = trials[0]
        for r in runs:
            print_summary(r, cfg.final_k)
        print_ablation(runs, cfg.final_k)
        if repeat > 1:
            click.echo(f"determinism over {repeat} runs (device={device}):")
            for i, m in enumerate(("dense", "bm25", "hybrid")):
                rc = [t[i]["recall"] for t in trials]
                mr = [t[i]["mrr"] for t in trials]
                flag = "" if max(rc) - min(rc) < 1e-9 else "  <-- UNSTABLE"
                click.echo(f"  {m:7} recall {min(rc):.2%}..{max(rc):.2%}  "
                           f"MRR {min(mr):.3f}..{max(mr):.3f}{flag}")
        return

    print_summary(score_run(data, index, cfg, mode, verbose=True), cfg.final_k)


if __name__ == "__main__":
    main()