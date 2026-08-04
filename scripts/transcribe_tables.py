#!/usr/bin/env python3
"""Transcribe table crops to markdown with a local VLM.

WHAT THIS IS TESTING: the PDF text layer destroys table structure. Row and
column headers survive as text, but the binding between them does not —
PhysBench Table 1 extracts as a run of collapsed headers followed by bare ✓/✗
sequences. `answerable@5` measured that as a 33pp gap: 83% of table questions
find the right paper, 50% receive the data.

An image of the table still HAS the binding, visually. So the question is
whether a vision model can put it back.

THE PROMPT MATTERS MORE THAN THE MODEL, which was not obvious. A first attempt
recovered every column header and every number and **dropped the row-label
column entirely**, leaving anonymous rows:

    | Property | Attribute | ... | Size |
    | Y        | N         | ... | 300,000 |     <- 300,000 of WHAT?

The leftmost column usually has no header in a scientific table, so the model
treated it as decoration. Naming it explicitly recovered CLEVRER, Cater,
CRIPP-VQA and the rest. That single instruction is the difference between a
transcription that answers "which benchmarks cover fluid interactions" and one
that cannot.

Resumable: re-running only transcribes what is missing.

    .venv/bin/python scripts/transcribe_tables.py
    .venv/bin/python scripts/transcribe_tables.py --limit 20 --model qwen2.5vl:7b
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import requests
from tqdm import tqdm

from arxiv_rag.config import Config

OLLAMA = "http://localhost:11434"

PROMPT = (
    "Transcribe this table as GitHub-flavoured markdown.\n"
    "CRITICAL: the leftmost column contains ROW LABELS (names of methods, "
    "datasets, or models) and often has no header of its own. You MUST include "
    "it as the first column, labelled 'Name'. Every data row must begin with "
    "its name.\n"
    "Use Y for a green check or tick and N for a red cross. Keep every numeric "
    "value exactly as printed, including commas. Output only the markdown table."
)


def transcribe(img: Path, model: str, host: str, timeout: int = 300) -> str:
    b64 = base64.b64encode(img.read_bytes()).decode()
    r = requests.post(
        f"{host}/api/chat",
        json={"model": model,
              "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
              "stream": False, "options": {"temperature": 0.0}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


@click.command()
@click.option("--model", default="qwen2.5vl:7b")
@click.option("--limit", type=int, default=None, help="Only do N tables.")
@click.option("--papers", default=None,
              help="Comma-separated arXiv ids to do FIRST. The manifest is in "
                   "sorted order, so without this an early read can be vacuous: "
                   "the eval papers sit late in the list and none of their "
                   "tables get transcribed for the first hour.")
@click.option("--host", default=OLLAMA)
def main(model, limit, papers, host):
    cfg = Config()
    fig_dir = cfg.data_dir / "figures"
    manifest = json.loads((fig_dir / "manifest.json").read_text())
    tables = [f for f in manifest if f["kind"] == "table"]

    out_path = fig_dir / "tables_md.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}

    todo = [t for t in tables if t["figure_id"] not in done]
    if papers:
        want = {p.strip().split("v")[0] for p in papers.split(",")}
        todo = [t for t in todo if t["arxiv_id"].split("v")[0] in want]
    if limit:
        todo = todo[:limit]

    click.echo(f"{len(tables)} tables · {len(done)} already done · "
               f"{len(todo)} to do · model={model}")
    if not todo:
        return

    times, failed = [], 0
    for t in tqdm(todo, desc="transcribing"):
        img = Path(t["image_path"])
        if not img.exists():
            failed += 1
            continue
        t0 = time.perf_counter()
        try:
            md = transcribe(img, model, host)
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            click.echo(f"\n  {t['figure_id']}: {type(exc).__name__}")
            continue
        times.append(time.perf_counter() - t0)
        done[t["figure_id"]] = {"model": model, "markdown": md}
        # Persist as we go: a 90-minute run that loses everything to one
        # exception at minute 88 is how the LoRA checkpoints were nearly lost.
        out_path.write_text(json.dumps(done, indent=2))

    times.sort()
    if times:
        click.echo(f"\n{len(times)} transcribed · median {times[len(times)//2]:.1f}s "
                   f"· total {sum(times)/60:.1f} min · {failed} failed")
    click.echo(f"wrote {out_path}")


if __name__ == "__main__":
    main()
