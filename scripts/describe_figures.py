#!/usr/bin/env python3
"""Generate VLM descriptions for extracted figures, resumably.

Writes into `data/figures/descriptions.json`, keyed by figure_id, and SKIPS
anything already present — so an interrupted run costs only the figures it had
not reached. On a ~660-figure corpus at a few seconds each, that resumability is
the difference between a nuisance and losing an hour.

The manifest is never mutated here. `scripts/eval_figures.py` joins the two, so
the caption-only corpus stays intact and the caption-vs-caption+description
ablation remains runnable at any time.

Usage:
    .venv/bin/python scripts/describe_figures.py --limit 20     # sample first
    .venv/bin/python scripts/describe_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from tqdm import tqdm

from arxiv_rag.config import Config
from arxiv_rag.describe import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    VLMUnavailable,
    check_model,
    describe_figure,
    load_cache,
    save_cache,
)


@click.command()
@click.option("--limit", type=int, default=None,
              help="Describe only N un-described figures. Use this first.")
@click.option("--model", default=DEFAULT_MODEL)
@click.option("--host", default=DEFAULT_HOST)
@click.option("--save-every", type=int, default=20,
              help="Flush the cache this often, so a crash loses little.")
def main(limit, model, host, save_every):
    cfg = Config()
    fig_dir = cfg.data_dir / "figures"
    manifest_path = fig_dir / "manifest.json"
    cache_path = fig_dir / "descriptions.json"

    if not manifest_path.exists():
        sys.exit("no manifest — run scripts/ingest_figures.py first")

    # Fail before the loop, with a message that names the real cause.
    try:
        check_model(model, host)
    except VLMUnavailable as exc:
        sys.exit(f"\n{exc}\n")

    manifest = json.loads(manifest_path.read_text())
    cache = load_cache(cache_path)

    todo = [f for f in manifest if f["figure_id"] not in cache]
    if limit:
        todo = todo[:limit]

    click.echo(f"{len(manifest)} figures · {len(cache)} already described · "
               f"{len(todo)} to do · model={model}")
    if not todo:
        return

    failed = []
    times = []
    for i, f in enumerate(tqdm(todo, desc="describing"), 1):
        try:
            d = describe_figure(f["image_path"], model=model, host=host)
        except Exception as exc:                          # noqa: BLE001
            # One bad crop must not end the run; record and move on.
            failed.append((f["figure_id"], f"{type(exc).__name__}: {exc}"[:120]))
            continue
        cache[d.figure_id] = {"text": d.text, "model": d.model, "seconds": d.seconds}
        times.append(d.seconds)
        if i % save_every == 0:
            save_cache(cache_path, cache)

    save_cache(cache_path, cache)

    if times:
        times.sort()
        click.echo(f"\n{len(times)} described · median {times[len(times)//2]:.1f}s "
                   f"· total {sum(times)/60:.1f} min")
    if failed:
        click.echo(f"FAILED {len(failed)}:")
        for fid, err in failed[:5]:
            click.echo(f"  {fid}: {err}")
    click.echo(f"wrote {cache_path}")


if __name__ == "__main__":
    main()
