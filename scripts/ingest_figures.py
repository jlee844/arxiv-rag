#!/usr/bin/env python3
"""Extract figures from every indexed paper and write a figure manifest.

Deliberately SEPARATE from `scripts/ingest.py`, and deliberately NOT indexing
anything by itself. Two reasons:

  1. Extraction is idempotent and cheap; indexing changes retrieval for every
     existing query. Keeping them apart means the figure corpus can be built and
     eyeballed before it is allowed near the index.
  2. The interesting question is whether adding ~600 figure chunks to a
     3288-chunk index HELPS figure queries more than it HURTS the 97 text cases.
     That is measured by `scripts/eval_figures.py`, which needs the manifest to
     exist first.

Usage:
    .venv/bin/python scripts/ingest_figures.py            # all indexed papers
    .venv/bin/python scripts/ingest_figures.py --limit 20
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from tqdm import tqdm

from arxiv_rag.config import Config
from arxiv_rag.figures import extract_figures


@click.command()
@click.option("--limit", type=int, default=None, help="Only process N papers.")
@click.option("--dpi", type=int, default=150,
              help="Render resolution. 150 keeps axis labels VLM-legible.")
@click.option("--index", "do_index", is_flag=True,
              help="Also add figure CAPTIONS to the live index. Off by "
                   "default: this changes retrieval for every existing "
                   "query, so run scripts/eval_figures.py first.")
def main(limit, dpi, do_index):
    cfg = Config()
    pdf_dir = cfg.data_dir / "pdfs"
    out_dir = cfg.data_dir / "figures"
    manifest_path = out_dir / "manifest.json"

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        sys.exit(f"no PDFs in {pdf_dir}")

    click.echo(f"{len(pdfs)} PDFs -> {out_dir}")

    all_figs: list[dict] = []
    per_paper = Counter()
    failed: list[tuple[str, str]] = []

    for pdf in tqdm(pdfs, desc="extracting"):
        aid = pdf.stem
        try:
            figs = extract_figures(pdf, aid, out_dir, dpi=dpi)
        except Exception as exc:                          # noqa: BLE001
            # One malformed PDF must not lose the whole run. This corpus
            # already contains files that emit MuPDF colorspace errors.
            failed.append((aid, f"{type(exc).__name__}: {exc}"))
            continue
        per_paper[aid] = len(figs)
        all_figs.extend(f.to_dict() for f in figs)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(all_figs, indent=2) + "\n")

    zero = sum(1 for p in pdfs if per_paper[p.stem] == 0)
    inks = sorted(f["ink"] for f in all_figs)
    click.echo(f"\n{len(all_figs)} figures from {len(pdfs) - zero}/{len(pdfs)} papers")
    click.echo(f"  papers yielding zero figures: {zero}  "
               f"(caption-anchored extraction; no caption => no figure)")
    if inks:
        click.echo(f"  ink fraction  min={inks[0]:.3f}  "
                   f"median={inks[len(inks)//2]:.3f}  max={inks[-1]:.3f}")
    click.echo(f"  most figures: {per_paper.most_common(3)}")
    if failed:
        click.echo(f"\nFAILED {len(failed)} PDFs:")
        for aid, err in failed[:5]:
            click.echo(f"  {aid}: {err[:90]}")
    click.echo(f"\nwrote {manifest_path}")

    if not do_index:
        click.echo("\nnot indexed (use --index). Measure first: "
                   "scripts/eval_figures.py")
        return

    # Captions only. VLM descriptions were measured and REJECTED — appending
    # them dropped figure recall@5 from 100% to 85.7% by diluting a short,
    # precise caption with generic prose (EVAL.md). The manifest keeps them so
    # the negative result stays reproducible; the index does not.
    from arxiv_rag.index import PaperIndex
    from arxiv_rag.parse import Chunk

    index = PaperIndex(cfg)
    before = index.count()

    # ORDER MATTERS. Delete existing figure chunks FIRST, then compute which
    # papers the TEXT index knows about.
    #
    # Getting this backwards made the orphan filter a silent no-op: figure
    # chunks from a previous run were already in the index, so their arxiv_ids
    # appeared in `indexed_papers()` and every orphan validated itself.
    stale = [f["figure_id"] for f in all_figs]
    existing = set(index._col.get(ids=stale)["ids"])
    if existing:
        index._col.delete(ids=list(existing))
        index._corpus = [c for c in index._corpus if c["chunk_id"] not in existing]
        click.echo(f"replacing {len(existing)} existing figure chunks")

    # Only index figures whose PAPER is already in the text index. Extraction
    # walks every PDF on disk, and ~20 of them were never ingested into Chroma —
    # indexing their figures silently grew the corpus from 115 to 132 "papers"
    # and made figures retrievable for documents whose text is not.
    known = set(index.indexed_papers())
    orphans = [f for f in all_figs if f["arxiv_id"] not in known]
    all_figs = [f for f in all_figs if f["arxiv_id"] in known]
    if orphans:
        click.echo(f"skipping {len(orphans)} figures from "
                   f"{len({o['arxiv_id'] for o in orphans})} papers whose text "
                   f"is not indexed")

    chunks = [
        Chunk(chunk_id=f["figure_id"], arxiv_id=f["arxiv_id"], title="",
              authors="", published="", section=f["label"],
              text=f"{f['label']}: {f['caption']}", chunk_index=i)
        for i, f in enumerate(all_figs)
    ]
    with index.batch():
        added = index.add_chunks(chunks)
    click.echo(f"indexed {added} figure captions  ({before} -> {index.count()} chunks)")


if __name__ == "__main__":
    main()
