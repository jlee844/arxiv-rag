#!/usr/bin/env python3
"""Table retrieval: findability vs answerability.

WHY TWO METRICS, AND WHY THE SECOND ONE IS THE REAL TEST:

Table queries score recall@5 83% against the shipped index, which looks fine
until you read what was retrieved. For *"which physical reasoning benchmarks
cover fluid interactions"*, the top hit is prose that merely **mentions** the
table:

    "We summarize the key features of these various benchmarks and compare
     them against our benchmark in Table 1."

The paper is correct. The answer is not there. Retrieval found the table by
proxy — via the sentence pointing at it — while the table's own contents
contributed nothing. A user gets a pointer, not the numbers.

So recall@5 measures FINDABILITY and systematically overstates how well tables
work. This script adds ANSWERABILITY: each case carries `answer_anchors`,
verified cell values that appear in the table body and essentially nowhere else
(a dataset size, a score, a model name in a results row). If a retrieved chunk
contains an anchor, the data actually arrived. If it only carries the caption or
surrounding prose, it did not.

    answerable@5 = the fraction of cases where >=1 anchor appears in the top-5

This is the number a table-parsing model (e.g. PaddleOCR-VL, which emits
structured markdown) would have to move. recall@5 is close to saturated and
cannot show that work succeeding.

Usage:
    .venv/bin/python scripts/eval_tables.py
    .venv/bin/python scripts/eval_tables.py --show-miss
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals" / "frameworks"))

import click

from arxiv_rag.config import Config
from arxiv_rag.index import PaperIndex
from arxiv_rag.retrieve import retrieve
from score import score          # noqa: E402

CASES = Path(__file__).parent.parent / "evals" / "table_cases.json"


@click.command()
@click.option("--k", default=5, help="Depth for both metrics.")
@click.option("--show-miss", is_flag=True, help="Print the top hit for failures.")
def main(k, show_miss):
    cases = json.loads(CASES.read_text())
    cfg = Config()
    index = PaperIndex(cfg)

    hits_by_case: dict[str, list] = {}

    def run(query: str):
        out = retrieve(query, index, cfg)
        hits_by_case[query] = out
        return out

    res = score(cases, run, k=k)

    click.echo(f"\ntable cases  n={res['n']}  (corpus {index.count()} chunks)")
    click.echo(f"\n{'metric':<26}{'value':>10}")
    click.echo(f"{'recall@'+str(k)+' (findability)':<26}{res['recall']:>10.4f}")
    click.echo(f"{'MRR':<26}{res['mrr']:>10.4f}")

    # Answerability — only over cases with verified anchors.
    scored = [c for c in cases if c.get("answer_anchors")]
    answerable = 0
    rows = []
    for c in scored:
        hits = hits_by_case.get(c["query"], [])[:k]
        blob = " ".join(h["text"] for h in hits)
        found = [a for a in c["answer_anchors"] if a in blob]
        answerable += bool(found)
        rows.append((c["id"], bool(found), found, hits))

    click.echo(f"{'answerable@'+str(k):<26}{answerable/max(len(scored),1):>10.4f}"
               f"   ({answerable}/{len(scored)} anchored cases)")

    click.echo(f"\n{'case':<36}{'found?':>8}{'answer?':>9}")
    by_id = {r["id"]: r for r in res["per_case"]}
    for cid, ok, found, _ in rows:
        f = "yes" if by_id[cid]["chunk_rank"] else "NO"
        a = "yes" if ok else "NO"
        click.echo(f"{cid:<36}{f:>8}{a:>9}")

    unscored = [c["id"] for c in cases if not c.get("answer_anchors")]
    if unscored:
        click.echo(f"\nnot scored for answerability (no verified anchors): "
                   f"{', '.join(unscored)}")

    if show_miss:
        click.echo("\n--- cases found but NOT answerable: what came back instead ---")
        for cid, ok, _, hits in rows:
            if ok or not by_id[cid]["chunk_rank"] or not hits:
                continue
            t = " ".join(hits[0]["text"].split())
            click.echo(f"\n{cid}\n  {hits[0]['arxiv_id']}  {t[:240]}")

    click.echo(
        "\nRead this as: retrieval finds the right PAPER for table questions, but "
        "the table's own values usually do not reach the top-5 — the match comes "
        "from prose referencing the table. That gap is what a structured table "
        "parser would close, and recall@5 alone cannot show it."
    )


if __name__ == "__main__":
    main()
