#!/usr/bin/env python3
"""Reclaim disk by deleting source PDFs that have already been ingested.

MEASURED, which is why this exists at all:

    data/pdfs       701 MB   78%   <- never touched at query time
    data/figures    107 MB   12%
    data/chroma_db   61 MB    7%
    data/bm25        12 MB    1%

Retrieval reads Chroma and the BM25 pickle. Generation reads retrieved chunk
text. **Nothing in the query path opens a PDF.** The 701 MB is build input that
stays resident forever by default.

WHY NOT "OCR THE PDFS TO SAVE SPACE": rendering a PDF to images makes it
BIGGER, not smaller. A text-layer PDF is compressed vector and text streams;
664 figure crops covering a fraction of total page area already cost 107 MB, so
rasterising every page would run to several GB. The saving comes from deleting
build input, not from re-encoding it.

WHAT YOU LOSE, stated plainly — a pruned PDF must be re-downloaded to:
  - re-chunk (changing chunk_size, or the section parser)
  - re-extract figures (scripts/ingest_figures.py)
  - verify a quote against the original document

Recovery is `scripts/ingest.py --ids <...>`; arXiv ids are stable and fetch.py
caches. But arXiv's ToS rate limit means ~135 papers is minutes, not seconds.

SAFETY: dry-run is the DEFAULT. Deleting requires --yes, and the script refuses
outright if the index looks empty or if a PDF has no chunks in the index —
because then the PDF is the only copy of that content and deleting it is
destructive, not a cache eviction.

Usage:
    .venv/bin/python scripts/prune_pdfs.py              # report only
    .venv/bin/python scripts/prune_pdfs.py --yes        # actually delete
    .venv/bin/python scripts/prune_pdfs.py --keep-figures-source --yes
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from arxiv_rag.config import Config
from arxiv_rag.index import PaperIndex


def _mb(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB"


@click.command()
@click.option("--yes", is_flag=True,
              help="Actually delete. Without this the script only reports.")
@click.option("--keep-figures-source", is_flag=True,
              help="Keep PDFs that have no extracted figures yet, so figure "
                   "extraction can still be run or re-run for them.")
def main(yes, keep_figures_source):
    cfg = Config()
    pdf_dir = cfg.data_dir / "pdfs"
    index = PaperIndex(cfg)

    n_chunks = index.count()
    if n_chunks == 0:
        sys.exit("REFUSING — the index is empty. The PDFs are the only copy of "
                 "this corpus; deleting them would lose it. Run scripts/ingest.py "
                 "first.")

    indexed = {a.split("v")[0] for a in index.indexed_papers()}

    figured: set[str] = set()
    manifest = cfg.data_dir / "figures" / "manifest.json"
    if manifest.exists():
        import json
        figured = {f["arxiv_id"].split("v")[0]
                   for f in json.loads(manifest.read_text())}

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    prunable, kept = [], []
    for p in pdfs:
        base = p.stem.split("v")[0]
        if base not in indexed:
            # This is the 19-PDF gap: downloaded but never ingested. Its content
            # exists NOWHERE else, so pruning it destroys data rather than
            # freeing a cache.
            kept.append((p, "not indexed — content exists only here"))
        elif keep_figures_source and base not in figured:
            kept.append((p, "no figures extracted yet"))
        else:
            prunable.append(p)

    total = sum(p.stat().st_size for p in pdfs)
    free = sum(p.stat().st_size for p in prunable)
    held = sum(p.stat().st_size for p, _ in kept)

    click.echo(f"index: {n_chunks} chunks / {len(indexed)} papers")
    click.echo(f"PDFs:  {len(pdfs)} files, {_mb(total)}\n")
    click.echo(f"  prunable (ingested): {len(prunable):>4}  {_mb(free)}")
    click.echo(f"  kept:                {len(kept):>4}  {_mb(held)}")

    reasons: dict[str, int] = {}
    for _, why in kept:
        reasons[why] = reasons.get(why, 0) + 1
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        click.echo(f"      {n:>3} x {why}")

    if not yes:
        click.echo(f"\nDRY RUN — nothing deleted. Would free {_mb(free)} "
                   f"({free/max(total,1):.0%} of the PDF directory).")
        click.echo("Re-run with --yes to delete. Recovery: "
                   "scripts/ingest.py --ids <arxiv_ids>")
        return

    for p in prunable:
        p.unlink()
    click.echo(f"\ndeleted {len(prunable)} PDFs, freed {_mb(free)}")
    click.echo("re-download with scripts/ingest.py --ids <arxiv_ids> when "
               "re-chunking or re-extracting figures")


if __name__ == "__main__":
    main()
