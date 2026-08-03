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
def main(limit, dpi):
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


if __name__ == "__main__":
    main()
