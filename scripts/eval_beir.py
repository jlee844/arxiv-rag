#!/usr/bin/env python3
"""Evaluate the SHIPPED retrieval stack on a BEIR dataset.

WHY THIS EXISTS: EVAL.md's headline numbers (recall@5 98.97%, MRR 0.940) come
from 145 cases written by the same agent that tuned the retriever. Honestly
computed, and not defensible to a skeptic -- cases written by the tuner encode
assumptions the retriever already satisfies. This scores the same code against
someone else's queries, someone else's corpus, and someone else's metric.

IT RUNS THE REAL STACK. `retrieve_bm25`, `retrieve_dense` and `retrieve` are
imported from arxiv_rag, not reimplemented here. A reimplementation would be a
different system wearing the same name, and the whole point is that this number
is about the shipped code.

SCORING IS pytrec_eval, NOT A HAND-ROLLED SCORER. This project has already
shipped a scorer whose docstring disagreed with its behaviour (chunk-level vs
paper-level MRR). The exercise is to stop grading my own homework, so the metric
must not be another thing to take on trust.

EXPECT WORSE NUMBERS. Published BM25 on SciFact is around nDCG@10 0.66. A hybrid
landing near that is respectable. **If this harness reports something far ABOVE
published BM25, that is a bug signal, not a win** -- most likely relevance
leakage. The gate is plausibility, not maximisation.

    .venv/bin/python scripts/eval_beir.py --dataset scifact
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from tqdm import tqdm

ROOT = Path(__file__).parent.parent

# Reference points from the BEIR paper, quoted for orientation only. They are NOT
# a target to beat and NOT recomputed here; a number far above them means a bug.
PUBLISHED = {
    "scifact":  {"BM25": 0.665, "note": "BEIR paper, nDCG@10"},
    "nfcorpus": {"BM25": 0.325, "note": "BEIR paper, nDCG@10"},
    "scidocs":  {"BM25": 0.158, "note": "BEIR paper, nDCG@10"},
}


def _load(dataset: str):
    d = ROOT / "data" / "beir" / dataset
    if not d.exists():
        raise SystemExit(f"{d} missing. Run:\n"
                         f"  .venv-beir/bin/python scripts/fetch_beir.py "
                         f"--dataset {dataset}")
    corpus = [json.loads(l) for l in (d / "corpus.jsonl").read_text().splitlines()]
    queries = {json.loads(l)["_id"]: json.loads(l)["text"]
               for l in (d / "queries.jsonl").read_text().splitlines()}
    qrels: dict[str, dict[str, int]] = {}
    for line in (d / "qrels.tsv").read_text().splitlines():
        q, c, s = line.split("\t")
        qrels.setdefault(q, {})[c] = int(s)
    return corpus, queries, qrels


@click.command()
@click.option("--dataset", default="scifact")
@click.option("--depth", default=100, help="Retrieval depth; recall@100 needs 100.")
@click.option("--limit", type=int, default=None, help="Only N judged queries (smoke).")
@click.option("--rebuild", is_flag=True, help="Force re-embed even if cached.")
def main(dataset, depth, limit, rebuild):
    import pytrec_eval

    from arxiv_rag.config import Config
    from arxiv_rag.index import PaperIndex
    from arxiv_rag.parse import Chunk
    from arxiv_rag.retrieve import retrieve, retrieve_bm25, retrieve_dense

    corpus, queries, qrels = _load(dataset)
    # Only judged queries are scorable. BEIR ships far more queries than qrels
    # (scifact: 1109 queries, 300 judged); scoring the rest silently averages in
    # zeros and understates every arm equally, which looks like a valid
    # comparison and is not.
    qids = [q for q in qrels if q in queries]
    if limit:
        qids = qids[:limit]
    click.echo(f"{dataset}: {len(corpus)} docs · {len(qids)} judged queries")

    # PERSISTENT per-dataset index, not a tempdir.
    #
    # The tempdir version re-embedded 64k documents on every run and threw the
    # result away -- then LitSearch crashed on a Chroma batch limit AFTER the
    # embedding pass, burning the expensive minutes and discarding them. Two
    # separate wastes: no reuse across runs, and no reuse after a late failure.
    #
    # Cached under the dataset's own directory, so a re-run (new metric, new
    # arm, tuned fusion) costs seconds instead of minutes. Still never the live
    # arxiv-rag index -- BEIR docs must not enter the shipped corpus.
    idx_dir = ROOT / "data" / "beir" / dataset / "index"
    if rebuild and idx_dir.exists():
        shutil.rmtree(idx_dir)
    cached = (idx_dir / "chroma").exists() and (idx_dir / "bm25").exists()
    cfg = Config()
    cfg.chroma_dir, cfg.bm25_dir = idx_dir / "chroma", idx_dir / "bm25"
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    cfg.bm25_dir.mkdir(parents=True, exist_ok=True)
    cfg.top_k = depth
    cfg.final_k = depth
    cfg.rerank = False
    cfg.query_expansion = None

    index = PaperIndex(cfg)
    chunks = [Chunk(chunk_id=d["_id"], arxiv_id=d["_id"],
                    title=d.get("title", ""), authors="", published="",
                    section="", text=(d.get("title", "") + "\n" +
                                      d.get("text", "")).strip(),
                    chunk_index=0)
              for d in corpus]
    # Chroma caps a single add() at 41,666 records. LitSearch (64k) and
    # BRIGHT (57k) both exceed it, and the failure is a hard ValueError AFTER
    # the embedding pass has already run -- so it burns the expensive minutes
    # first and then throws away the result. Slice the add instead.
    BATCH = 20000
    t0 = time.perf_counter()
    if cached and index.count() >= len(chunks) * 0.99:
        click.echo(f"reusing cached index ({index.count()} docs) — no re-embed")
    else:
        with index.batch():
            for i in range(0, len(chunks), BATCH):
                index.add_chunks(chunks[i:i + BATCH])
        click.echo(f"indexed {index.count()} docs in {time.perf_counter()-t0:.0f}s")

    arms = {
        "bm25":   lambda q: retrieve_bm25(q, index, cfg),
        "dense":  lambda q: retrieve_dense(q, index, cfg),
        "hybrid": lambda q: retrieve(q, index, cfg),
    }
    evaluator = pytrec_eval.RelevanceEvaluator(
        {q: qrels[q] for q in qids}, {"ndcg_cut.10", "recall_100", "map"})

    results = {}
    for name, fn in arms.items():
        run: dict[str, dict[str, float]] = {}
        t0 = time.perf_counter()
        for qid in tqdm(qids, desc=name):
            hits = fn(queries[qid])
            # Rank by position: the retrievers return an ordered list, and
            # their internal scores are not comparable across arms (RRF ranks
            # are not similarities). Position is the only common currency.
            run[qid] = {h["arxiv_id"]: float(len(hits) - i)
                        for i, h in enumerate(hits[:depth])}
        took = time.perf_counter() - t0
        scores = evaluator.evaluate(run)
        agg = {m: sum(s[m] for s in scores.values()) / max(len(scores), 1)
               for m in ("ndcg_cut_10", "recall_100", "map")}
        agg["ms_per_query"] = 1000 * took / max(len(qids), 1)
        results[name] = agg

    click.echo("\n" + "=" * 66)
    click.echo(f"{dataset}  n={len(qids)} judged queries")
    click.echo(f"{'arm':<10}{'nDCG@10':>10}{'recall@100':>13}{'MAP':>9}{'ms/query':>11}")
    click.echo("-" * 66)
    for name, a in results.items():
        click.echo(f"{name:<10}{a['ndcg_cut_10']:>10.4f}{a['recall_100']:>13.4f}"
                   f"{a['map']:>9.4f}{a['ms_per_query']:>11.1f}")
    ref = PUBLISHED.get(dataset)
    if ref:
        click.echo(f"{'[published BM25]':<10}{ref['BM25']:>10.3f}"
                   f"{'':>13}{'':>9}   {ref['note']}")

    out = ROOT / "evals" / f"beir_{dataset}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"dataset": dataset, "n_queries": len(qids),
                               "n_docs": len(corpus), "depth": depth,
                               "results": results,
                               "published_bm25": ref}, indent=2) + "\n")
    click.echo(f"\nwrote {out}")

    if ref:
        got = results["bm25"]["ndcg_cut_10"]
        if got > ref["BM25"] * 1.25:
            click.echo(f"\n!! BM25 nDCG@10 {got:.3f} is far ABOVE published "
                       f"{ref['BM25']:.3f}. Treat as a BUG (relevance leakage / "
                       f"id mismatch), not a result.")
        elif got < ref["BM25"] * 0.6:
            click.echo(f"\n!! BM25 nDCG@10 {got:.3f} is far BELOW published "
                       f"{ref['BM25']:.3f}. Check tokenisation and doc ids.")
        else:
            click.echo(f"\nBM25 is in a plausible band vs published "
                       f"{ref['BM25']:.3f} -- harness looks sane.")



if __name__ == "__main__":
    main()
