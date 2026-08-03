#!/usr/bin/env python3
"""Does indexing figure captions help — and what does it cost the text cases?

THE QUESTION THIS ANSWERS, and the reason it exists before any feature ships:

Adding 664 figure chunks to a 3288-chunk index grows it by 20%. Those chunks are
short, dense, and topically identical to the papers they came from, which is
exactly the profile of a distractor. So there are two effects and they point in
opposite directions:

    + figure queries  become answerable at all
    - text queries    may be displaced by caption chunks that merely look relevant

Reporting only the first would be marketing. This script measures both, on the
SAME 97 hand-written cases the rest of the repo is scored on, and treats a
regression there as disqualifying regardless of how good the figure numbers look.

SAFETY: the shipped index is never mutated. The augmented index is built into a
throwaway copy under `data/figures_index/`, so a bad result costs nothing and
the production retriever cannot be left in a half-modified state.

Usage:
    .venv/bin/python scripts/eval_figures.py
    .venv/bin/python scripts/eval_figures.py --keep    # leave the copy on disk
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals" / "frameworks"))

import click

from arxiv_rag.config import Config
from arxiv_rag.index import PaperIndex
from arxiv_rag.parse import Chunk
from arxiv_rag.retrieve import retrieve
from score import load_cases, score          # noqa: E402

FIG_CASES = Path(__file__).parent.parent / "evals" / "figure_cases.json"


def figure_chunks(manifest: list[dict]) -> list[Chunk]:
    """Turn manifest rows into Chunks so they travel the EXISTING index path.

    No new schema, no parallel store: a figure is a short text chunk whose
    `section` is its label ("Figure 3") and whose `chunk_id` is its figure_id.
    That means it is embedded, BM25-indexed, fused, and gated by exactly the
    same code as body text — so any measured difference is about the CONTENT
    being added, not about a second retrieval path behaving differently.

    The rendered PNG is recovered later by looking `chunk_id` up in the
    manifest, which keeps the image out of the index entirely.
    """
    out = []
    for i, f in enumerate(manifest):
        text = f"{f['label']}: {f['caption']}"
        if f.get("description"):
            text += "\n" + f["description"]
        out.append(Chunk(
            chunk_id=f["figure_id"],
            arxiv_id=f["arxiv_id"],
            title=f.get("title", ""),
            authors="",
            published=f.get("published", ""),
            section=f["label"],
            text=text,
            chunk_index=i,
        ))
    return out


def build_augmented(cfg: Config, manifest: list[dict], dest: Path) -> PaperIndex:
    """Copy the live index, then add figure chunks to the copy."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(cfg.chroma_dir, dest / "chroma_db")
    shutil.copytree(cfg.bm25_dir, dest / "bm25")

    aug = Config()
    aug.chroma_dir = dest / "chroma_db"
    aug.bm25_dir = dest / "bm25"
    index = PaperIndex(aug)
    with index.batch():
        index.add_chunks(figure_chunks(manifest))
    return index, aug


def run(index, cfg):
    def _run(query):
        return retrieve(query, index, cfg)
    return _run


@click.command()
@click.option("--keep", is_flag=True, help="Keep the augmented index on disk.")
@click.option("--with-descriptions", is_flag=True,
              help="Append VLM descriptions to captions. This is the Phase 2 "
                   "ablation: run it against the caption-only numbers to see "
                   "whether the VLM added anything the caption did not.")
def main(keep, with_descriptions):
    cfg = Config()
    manifest_path = cfg.data_dir / "figures" / "manifest.json"
    if not manifest_path.exists():
        sys.exit("no manifest — run scripts/ingest_figures.py first")
    manifest = json.loads(manifest_path.read_text())

    # Join VLM descriptions if they exist. Kept in a separate file so the
    # caption-only corpus is never overwritten and the ablation stays runnable.
    desc_path = cfg.data_dir / "figures" / "descriptions.json"
    n_desc = 0
    if with_descriptions and desc_path.exists():
        cache = json.loads(desc_path.read_text())
        for f in manifest:
            d = cache.get(f["figure_id"])
            if d and d.get("text"):
                f["description"] = d["text"]
                n_desc += 1
        click.echo(f"joined {n_desc} VLM descriptions")
    elif with_descriptions:
        click.echo("--with-descriptions requested but descriptions.json is absent; "
                   "run scripts/describe_figures.py (caption-only results follow)")

    base_index = PaperIndex(cfg)
    click.echo(f"baseline index: {base_index.count()} chunks")
    click.echo(f"manifest: {len(manifest)} figures")

    dest = cfg.data_dir / "figures_index"
    aug_index, aug_cfg = build_augmented(cfg, manifest, dest)
    click.echo(f"augmented index: {aug_index.count()} chunks "
               f"(+{aug_index.count() - base_index.count()})")

    pos, _ = load_cases()

    click.echo("\n" + "=" * 68)
    click.echo("A. REGRESSION CHECK — the 97 hand-written TEXT cases")
    click.echo("   A drop here disqualifies the feature no matter what B says.")
    base = score(pos, run(base_index, cfg))
    aug = score(pos, run(aug_index, aug_cfg))
    click.echo(f"\n{'':<12}{'recall@5':>10}{'MRR':>10}")
    click.echo(f"{'baseline':<12}{base['recall']:>10.4f}{base['mrr']:>10.4f}")
    click.echo(f"{'+figures':<12}{aug['recall']:>10.4f}{aug['mrr']:>10.4f}")
    click.echo(f"{'Δ':<12}{aug['recall']-base['recall']:>+10.4f}"
               f"{aug['mrr']-base['mrr']:>+10.4f}")

    click.echo(f"\n{'tag':<14}{'n':>4}{'base':>9}{'+figs':>9}{'Δ':>9}")
    for tag, d in sorted(base["by_tag"].items(), key=lambda kv: -kv[1]["n"]):
        a = aug["by_tag"][tag]
        click.echo(f"{tag:<14}{d['n']:>4}{d['mrr']:>9.3f}{a['mrr']:>9.3f}"
                   f"{a['mrr']-d['mrr']:>+9.3f}")

    # How often does a figure chunk displace a text chunk in the top 5?
    displaced = 0
    fig_ids = {f["figure_id"] for f in manifest}
    for c in pos:
        hits = retrieve(c["query"], aug_index, aug_cfg)
        displaced += sum(1 for h in hits[:5] if h["chunk_id"] in fig_ids)
    click.echo(f"\nfigure chunks appearing in top-5 of TEXT queries: "
               f"{displaced}/{len(pos)*5} slots ({displaced/(len(pos)*5):.1%})")

    click.echo("\n" + "=" * 68)
    click.echo("B. FIGURE QUERIES — can the figures be found at all?")
    if not FIG_CASES.exists():
        click.echo(f"   (no {FIG_CASES.name}; skipping)")
    else:
        fcases = json.loads(FIG_CASES.read_text())
        fb = score(fcases, run(base_index, cfg))
        fa = score(fcases, run(aug_index, aug_cfg))
        click.echo(f"\n{'':<12}{'recall@5':>10}{'MRR':>10}   (n={len(fcases)})")
        click.echo(f"{'baseline':<12}{fb['recall']:>10.4f}{fb['mrr']:>10.4f}")
        click.echo(f"{'+figures':<12}{fa['recall']:>10.4f}{fa['mrr']:>10.4f}")
        click.echo(f"{'Δ':<12}{fa['recall']-fb['recall']:>+10.4f}"
                   f"{fa['mrr']-fb['mrr']:>+10.4f}")

    if not keep:
        shutil.rmtree(dest)
        click.echo(f"\nremoved {dest} (use --keep to retain)")


if __name__ == "__main__":
    main()
