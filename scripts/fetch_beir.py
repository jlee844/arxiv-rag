#!/usr/bin/env python3
"""Download a BEIR dataset and dump it to plain files.

RUN WITH .venv-beir, NOT the main venv:

    .venv-beir/bin/python scripts/fetch_beir.py --dataset scifact

The main venv's pins are what produced every published number in EVAL.md. Adding
`datasets` (and its transitive pyarrow/fsspec constraints) to it risks moving
those numbers silently, which is exactly the failure this whole exercise exists
to prevent. So the download happens in an isolated venv and lands as plain JSONL,
which the main venv reads with the standard library.

Emits, under data/beir/<dataset>/:
    corpus.jsonl    {"_id", "title", "text"}
    queries.jsonl   {"_id", "text"}
    qrels.tsv       query-id \t corpus-id \t score      (TREC-style)
"""

from __future__ import annotations

import json
from pathlib import Path

import click

ROOT = Path(__file__).parent.parent

# BEIR on HF splits corpus/queries from qrels. The qrels repo name is not always
# "<name>-qrels", so it is listed explicitly rather than guessed.
SPECS = {
    "scifact":  {"repo": "BeIR/scifact",  "qrels": "BeIR/scifact-qrels",  "split": "test"},
    "nfcorpus": {"repo": "BeIR/nfcorpus", "qrels": "BeIR/nfcorpus-qrels", "split": "test"},
    "scidocs":  {"repo": "BeIR/scidocs",  "qrels": "BeIR/scidocs-qrels",  "split": "test"},
}


@click.command()
@click.option("--dataset", type=click.Choice(sorted(SPECS)), default="scifact")
def main(dataset):
    from datasets import load_dataset

    spec = SPECS[dataset]
    out = ROOT / "data" / "beir" / dataset
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_dataset(spec["repo"], "corpus", split="corpus")
    with (out / "corpus.jsonl").open("w") as f:
        for r in corpus:
            f.write(json.dumps({"_id": str(r["_id"]),
                                "title": r.get("title", "") or "",
                                "text": r.get("text", "") or ""}) + "\n")

    queries = load_dataset(spec["repo"], "queries", split="queries")
    with (out / "queries.jsonl").open("w") as f:
        for r in queries:
            f.write(json.dumps({"_id": str(r["_id"]),
                                "text": r.get("text", "") or ""}) + "\n")

    qrels = load_dataset(spec["qrels"], split=spec["split"])
    n = 0
    with (out / "qrels.tsv").open("w") as f:
        for r in qrels:
            f.write(f"{r['query-id']}\t{r['corpus-id']}\t{int(r['score'])}\n")
            n += 1

    # Only queries that HAVE judgements are scorable. BEIR ships far more
    # queries than qrels (scifact: ~1.1k queries, ~300 judged for test), and
    # scoring the unjudged ones silently averages in zeros.
    judged = {l.split("\t")[0] for l in (out / "qrels.tsv").read_text().splitlines()}
    click.echo(f"{dataset}:")
    click.echo(f"  corpus   {len(corpus):>7}")
    click.echo(f"  queries  {len(queries):>7}  (judged in {spec['split']}: {len(judged)})")
    click.echo(f"  qrels    {n:>7}")
    click.echo(f"  -> {out}")


if __name__ == "__main__":
    main()
