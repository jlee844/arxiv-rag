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
@click.option("--with-markdown", is_flag=True,
              help="Add VLM table transcriptions to a THROWAWAY index copy and "
                   "re-measure. The live index is never touched.")
def main(k, show_miss, with_markdown):
    cases = json.loads(CASES.read_text())
    cfg = Config()

    tmp = None
    if with_markdown:
        # Copy the index, exactly as eval_figures.py does. Measuring on a copy is
        # what makes this reversible: a technique gets adopted because a number
        # moved, not because it is already installed.
        import shutil
        import tempfile

        md_path = cfg.data_dir / "figures" / "tables_md.json"
        manifest = json.loads((cfg.data_dir / "figures" / "manifest.json").read_text())
        md = json.loads(md_path.read_text()) if md_path.exists() else {}
        if not md:
            raise SystemExit("no transcriptions; run scripts/transcribe_tables.py")

        tmp = Path(tempfile.mkdtemp(prefix="tbl_eval_"))
        shutil.copytree(cfg.chroma_dir, tmp / "chroma")
        shutil.copytree(cfg.bm25_dir, tmp / "bm25")
        cfg.chroma_dir, cfg.bm25_dir = tmp / "chroma", tmp / "bm25"

        from arxiv_rag.parse import Chunk
        index = PaperIndex(cfg)
        known = {a.split("v")[0] for a in index.indexed_papers()}
        by_id = {f["figure_id"]: f for f in manifest}
        chunks = []
        for fid, rec in md.items():
            f = by_id.get(fid)
            if not f or f["arxiv_id"].split("v")[0] not in known:
                continue
            chunks.append(Chunk(
                chunk_id=f"{fid}-md", arxiv_id=f["arxiv_id"], title="",
                authors="", published="", section=f["label"],
                # Caption FIRST so the chunk keeps the author's precise wording,
                # then the transcribed grid. The figure ablation showed that
                # burying a caption under generated prose costs recall.
                text=f"{f['label']}: {f['caption']}\n{rec['markdown']}",
                chunk_index=0))
        with index.batch():
            added = index.add_chunks(chunks)
        click.echo(f"+{added} transcribed tables into a throwaway index copy")
    else:
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
    #
    # ALSO records WHERE the table chunk landed, because `answerable@5` alone
    # conflates two failures that need opposite fixes:
    #
    #   table chunk retrieved, anchor absent -> STRUCTURE problem. The chunk is
    #       there and unreadable; a transcription fixes it.
    #   table chunk never retrieved          -> RANKING problem. Transcription
    #       cannot help, because the chunk never reaches the reader.
    #
    # The distinction is not cosmetic. The observed top hit for a
    # structure-dependent query was PROSE REFERENCING the table
    # ("...compare them against our benchmark in Table 1"), which is retrieval
    # behaving correctly: a well-formed sentence embeds far better than
    # "Property Attribute ... ✓ ✗ ✗ 300,000 ✗". If that is the whole story, the
    # fix is to resolve the reference, not to improve the ranking.
    scored = [c for c in cases if c.get("answer_anchors")]
    answerable = 0
    tbl_retrieved = 0
    rows = []
    for c in scored:
        hits = hits_by_case.get(c["query"], [])[:k]
        blob = " ".join(h["text"] for h in hits)
        found = [a for a in c["answer_anchors"] if a in blob]
        answerable += bool(found)

        # Did ANY table-derived chunk make the top-k, and at what rank?
        t_rank = None
        for i, h in enumerate(hits, 1):
            cid = str(h.get("chunk_id", ""))
            sect = str(h.get("section", ""))
            if "-table" in cid or sect.lower().startswith("table"):
                t_rank = i
                break
        tbl_retrieved += t_rank is not None
        rows.append((c["id"], bool(found), found, hits, t_rank))

    n_sc = max(len(scored), 1)
    click.echo(f"{'answerable@'+str(k):<26}{answerable/n_sc:>10.4f}"
               f"   ({answerable}/{len(scored)} anchored cases)")
    click.echo(f"{'table chunk in top-'+str(k):<26}{tbl_retrieved/n_sc:>10.4f}"
               f"   ({tbl_retrieved}/{len(scored)})")

    click.echo(f"\n{'case':<34}{'paper?':>8}{'tbl@':>6}{'answer?':>9}  diagnosis")
    by_id = {r["id"]: r for r in res["per_case"]}
    diag_counts = {"structure": 0, "ranking": 0, "ok": 0, "not found": 0}
    for cid, ok, found, _, t_rank in rows:
        f = "yes" if by_id[cid]["chunk_rank"] else "NO"
        a = "yes" if ok else "NO"
        if ok:
            d = "ok"
        elif f == "NO":
            d = "not found"
        elif t_rank is not None:
            d = "structure"          # table chunk arrived, data unreadable
        else:
            d = "ranking"            # table chunk never surfaced
        diag_counts[d] += 1
        click.echo(f"{cid:<34}{f:>8}{str(t_rank or '-'):>6}{a:>9}  {d}")

    click.echo(f"\nFAILURE SPLIT  structure={diag_counts['structure']}  "
               f"ranking={diag_counts['ranking']}  "
               f"paper-not-found={diag_counts['not found']}  ok={diag_counts['ok']}")
    if diag_counts["ranking"] > diag_counts["structure"]:
        click.echo("  -> RANKING dominates. A better table PARSER cannot fix this; "
                   "the table chunk never reaches the reader. Resolve the "
                   "'see Table N' reference instead of trying to outrank prose.")
    elif diag_counts["structure"] > 0:
        click.echo("  -> STRUCTURE dominates. The chunk arrives and is unreadable, "
                   "which is exactly what a transcription fixes.")

    unscored = [c["id"] for c in cases if not c.get("answer_anchors")]
    if unscored:
        click.echo(f"\nnot scored for answerability (no verified anchors): "
                   f"{', '.join(unscored)}")

    if show_miss:
        click.echo("\n--- cases found but NOT answerable: what came back instead ---")
        for cid, ok, _, hits, _t in rows:
            if ok or not by_id[cid]["chunk_rank"] or not hits:
                continue
            t = " ".join(hits[0]["text"].split())
            click.echo(f"\n{cid}\n  {hits[0]['arxiv_id']}  {t[:240]}")

    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        click.echo(f"\nremoved {tmp} (throwaway index copy)")

    click.echo(
        "\nRead this as: retrieval finds the right PAPER for table questions, but "
        "the table's own values usually do not reach the top-5 — the match comes "
        "from prose referencing the table. That gap is what a structured table "
        "parser would close, and recall@5 alone cannot show it."
    )


if __name__ == "__main__":
    main()
