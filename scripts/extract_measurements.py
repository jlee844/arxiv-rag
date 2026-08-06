#!/usr/bin/env python3
"""Transcribed table markdown -> measurement rows. P1 of the numeric-index spec.

A transcribed table is a grid of strings. A measurement is
(paper, method, dataset/metric, value). Turning one into the other is mostly
about deciding what is NOT a measurement, which is where the precision comes
from -- and precision is what matters here, because a wrong number does not fail
loudly. It sits in the index and silently poisons every query that touches it.

The gate in the spec is >=90% precision with recall merely reported. So every
rule below is biased toward dropping a row rather than guessing at it.

WHAT IS DROPPED, AND WHY:
  * non-numeric cells                    not a measurement
  * the row-label column                 it is the key, not a value
  * rows whose label is not an entity    "total", "average", "ours", "human",
                                         "random" -- these are aggregates or
                                         placeholders, and joining on them
                                         across papers produces nonsense
                                         ("random" appears in 5 papers and means
                                         something different in each)
  * values outside plausible ranges      a percentage of 4000 is a parse error
  * cells with >1 number ("72.3/68.1")   ambiguous without knowing the split;
                                         recorded as unresolved rather than
                                         guessed

Emits data/measurements.json and prints the precision-relevant diagnostics.

    .venv/bin/python scripts/extract_measurements.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

ROOT = Path(__file__).parent.parent

# Labels that look like entities but are not joinable across papers.
NON_ENTITY = {
    "total", "average", "avg", "mean", "overall", "all", "sum",
    "ours", "our method", "our model", "baseline", "random", "human",
    "others", "other", "none", "-", "n/a", "na", "chance", "majority",
    "random baseline", "human expert", "std", "sd", "count", "number",
}

_NUM = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$|^[+-]?\d*\.?\d+%?$")
_MULTI = re.compile(r"\d.*[/±].*\d")


def _clean(s: str) -> str:
    s = re.sub(r"[*_`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _as_number(cell: str):
    """Return (value, is_percent) or None. Deliberately strict."""
    c = _clean(cell)
    if not c or c in {"-", "--", "n/a", "N/A", "—"}:
        return None
    if _MULTI.search(c):
        return "AMBIGUOUS"
    pct = c.endswith("%")
    c2 = c.rstrip("%").replace(",", "")
    if not _NUM.match(c.replace(",", "")):
        return None
    try:
        v = float(c2)
    except ValueError:
        return None
    return (v, pct)


def _parse_table(md: str):
    """Return (headers, rows) from GitHub-flavoured markdown."""
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    lines = [l for l in lines if set(_clean(l).replace("|", "").strip()) - set("-: ")]
    if len(lines) < 2:
        return None, []
    headers = [_clean(c) for c in lines[0].strip("|").split("|")]
    rows = []
    for l in lines[1:]:
        cells = [_clean(c) for c in l.strip("|").split("|")]
        if len(cells) < 2:
            continue
        rows.append(cells)
    return headers, rows


@click.command()
@click.option("--out", default="data/measurements.json")
def main(out):
    fd = ROOT / "data" / "figures"
    md = json.loads((fd / "tables_md.json").read_text())
    man = {f["figure_id"]: f for f in json.loads((fd / "manifest.json").read_text())}

    measurements = []
    drop = Counter()
    tables_used = 0

    for fid, rec in md.items():
        meta = man.get(fid)
        if not meta:
            drop["no manifest entry"] += 1
            continue
        aid = meta["arxiv_id"].split("v")[0]
        headers, rows = _parse_table(rec["markdown"])
        if not headers or not rows:
            drop["unparseable table"] += 1
            continue
        tables_used += 1

        for cells in rows:
            label = cells[0]
            lab_norm = re.sub(r"\s*\([^)]*\)", "", label).strip().lower()
            lab_norm = re.sub(r"[^a-z0-9.\- ]", "", lab_norm).strip()
            if not lab_norm:
                drop["empty row label"] += 1
                continue
            if lab_norm in NON_ENTITY:
                drop["non-entity row label"] += 1
                continue
            if _as_number(label) is not None:
                drop["numeric row label"] += 1
                continue

            for j, cell in enumerate(cells[1:], start=1):
                col = headers[j] if j < len(headers) else f"col{j}"
                got = _as_number(cell)
                if got is None:
                    drop["non-numeric cell"] += 1
                    continue
                if got == "AMBIGUOUS":
                    drop["ambiguous cell (a/b or +/-)"] += 1
                    continue
                v, pct = got
                if pct and not (0.0 <= v <= 100.0):
                    drop["percent out of range"] += 1
                    continue
                measurements.append({
                    "paper": aid, "method": lab_norm, "method_raw": label,
                    "column": col, "value": v, "is_percent": pct,
                    "source": f"table:{fid}",
                    "caption": " ".join(meta.get("caption", "").split())[:160],
                })

    p = ROOT / out
    p.write_text(json.dumps(measurements, indent=2) + "\n")

    click.echo(f"tables transcribed   : {len(md)}")
    click.echo(f"tables parsed        : {tables_used}")
    click.echo(f"MEASUREMENTS         : {len(measurements)}")
    click.echo(f"distinct methods     : {len({m['method'] for m in measurements})}")
    click.echo(f"distinct columns     : {len({m['column'] for m in measurements})}")
    click.echo(f"papers covered       : {len({m['paper'] for m in measurements})}")
    click.echo("\ndropped (precision comes from these):")
    for k, v in drop.most_common():
        click.echo(f"  {k:<32}{v:>7}")

    # The join question, now on canonicalised method names rather than raw cells.
    bym = defaultdict(set)
    for m in measurements:
        bym[m["method"]].add(m["paper"])
    multi = {k: v for k, v in bym.items() if len(v) >= 2}
    click.echo(f"\nmethods in >=2 papers: {len(multi)} / {len(bym)} "
               f"({len(multi)/max(len(bym),1):.1%})")
    click.echo("top joinable methods:")
    for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:15]:
        click.echo(f"  {k:<34}{len(v)} papers")
    click.echo(f"\nwrote {p}")


if __name__ == "__main__":
    main()
