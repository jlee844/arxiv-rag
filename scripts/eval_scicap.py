#!/usr/bin/env python3
"""Does a VLM description DILUTE a figure caption? Measured on SciCap.

THE CLAIM UNDER TEST. EVAL.md says VLM figure descriptions were measured and
rejected: figure recall 100% -> 85.7% with qwen2.5vl:7b, attributed to generated
prose diluting a short precise caption. That rests on **14 hand-written cases**,
which makes it the least defensible claim left in the repo now that retrieval has
outside validation.

WHAT SCICAP CAN AND CANNOT DO. SciCap ships arXiv figures with author captions
and no queries at all, so it cannot replicate a retrieval experiment directly.
Pretending otherwise would be the same self-authorship problem in new clothes.

What it CAN do is test the stated MECHANISM, which is the disputed part:

  1. DILUTION (lexical). If a description is mostly generic prose, adding it
     drops the share of the indexed text that is caption -- and drops the
     density of the rare, discriminative tokens retrieval actually keys on.
     No queries needed; this is a property of the text.

  2. KNOWN-ITEM RETRIEVAL (mechanical, no authored queries). Query = a
     TRUNCATED caption (first 60% of tokens). Corpus = every figure, indexed
     either caption-only or caption+description. If the description dilutes,
     recall drops. Truncation is what stops this being trivial exact match, and
     it is mechanical, so no query is written by me.

Neither is the original experiment. Both are outside data, and (1) is the exact
quantity the rejection claimed.

    .venv/bin/python scripts/eval_scicap.py --n 300
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "scicap"


def _local(n):
    """Dilution on arxiv-rag's OWN figures, using the cached Phase 2 descriptions.

    Not outside data -- but the claim being tested is about THIS corpus, and it
    currently rests on 14 hand-written cases. The same measurement at n=664 is
    47x the evidence on exactly the figures the rejection was about, and it
    costs nothing because the descriptions are already cached from the run that
    produced the rejection.

    SciCap would add outside figures; it also costs a 23.3 GB multi-part image
    download, so it is worth doing only if this disagrees with the claim.
    """
    fd = ROOT / "data" / "figures"
    desc = json.loads((fd / "descriptions.json").read_text())
    man = {f["figure_id"]: f for f in json.loads((fd / "manifest.json").read_text())}
    rows = []
    for fid, d in desc.items():
        m = man.get(fid)
        if not m or m.get("kind") != "figure":
            continue
        cap = (m.get("caption") or "").strip()
        text = d if isinstance(d, str) else (d.get("description") or d.get("text") or "")
        if len(cap) < 30 or len(text) < 30:
            continue
        rows.append({"caption": cap, "desc": text})
        if n and len(rows) >= n:
            break
    _report(rows, "local (arxiv-rag figures, cached Phase 2 descriptions)")
    return


def _report(rows, label):
    import collections
    from rank_bm25 import BM25Okapi

    click.echo(f"{len(rows)} figures — {label}")
    if not rows:
        raise SystemExit("no (caption, description) pairs found")

    df = collections.Counter()
    for r in rows:
        df.update(set(_toks(r["caption"])))
    n_docs = max(len(rows), 1)

    cap_share, rare_cap, rare_both, cap_len, desc_len = [], [], [], [], []
    for r in rows:
        c, d = _toks(r["caption"]), _toks(r["desc"])
        cap_len.append(len(c)); desc_len.append(len(d))
        cap_share.append(len(c) / max(len(c) + len(d), 1))
        rc = [t for t in c if df[t] / n_docs < 0.05]
        rare_cap.append(len(rc) / max(len(c), 1))
        rare_both.append(len(rc) / max(len(c) + len(d), 1))

    mean = lambda xs: sum(xs) / max(len(xs), 1)   # noqa: E731
    click.echo("\n=== 1. DILUTION (property of the text, no queries needed) ===")
    click.echo(f"  caption length (tokens)        {mean(cap_len):>8.1f}")
    click.echo(f"  description length (tokens)    {mean(desc_len):>8.1f}")
    click.echo(f"  caption share of indexed text  {mean(cap_share):>8.1%}")
    click.echo(f"  rare-token density, caption    {mean(rare_cap):>8.1%}")
    click.echo(f"  rare-token density, cap+desc   {mean(rare_both):>8.1%}")
    drop = 1 - mean(rare_both) / max(mean(rare_cap), 1e-9)
    click.echo(f"  -> discriminative density falls {drop:>7.1%} when the "
               f"description is added")

    def known_item(texts):
        corpus = [_toks(t) for t in texts]
        bm = BM25Okapi(corpus)
        h1 = h5 = 0
        for i, r in enumerate(rows):
            q = _toks(r["caption"])
            q = q[:max(3, int(len(q) * 0.6))]
            sc = bm.get_scores(q)
            order = sorted(range(len(corpus)), key=lambda j: -sc[j])
            h1 += order[0] == i
            h5 += i in order[:5]
        return h1 / len(rows), h5 / len(rows)

    a1, a5 = known_item([r["caption"] for r in rows])
    b1, b5 = known_item([f"{r['caption']}\n{r['desc']}" for r in rows])
    click.echo("\n=== 2. KNOWN-ITEM RETRIEVAL (query = first 60% of caption) ===")
    click.echo(f"{'index':<22}{'recall@1':>10}{'recall@5':>10}")
    click.echo(f"{'caption only':<22}{a1:>10.3f}{a5:>10.3f}")
    click.echo(f"{'caption + VLM desc':<22}{b1:>10.3f}{b5:>10.3f}")
    click.echo(f"{'delta':<22}{b1-a1:>+10.3f}{b5-a5:>+10.3f}")
    click.echo("\n  -> " + ("SUPPORTS the rejection: descriptions measurably hurt."
                            if b5 < a5 - 0.005 else
                            "DOES NOT support the rejection: no measurable drop."))

    p = ROOT / "evals" / "figure_dilution.json"
    p.write_text(json.dumps({
        "n": len(rows), "source": label,
        "caption_tokens": mean(cap_len), "desc_tokens": mean(desc_len),
        "caption_share": mean(cap_share),
        "rare_density_caption": mean(rare_cap),
        "rare_density_cap_desc": mean(rare_both),
        "known_item": {"caption_only": {"r@1": a1, "r@5": a5},
                       "caption_desc": {"r@1": b1, "r@5": b5}},
    }, indent=2) + "\n")
    click.echo(f"\nwrote {p}")

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")


def _toks(t: str) -> list[str]:
    return _WORD.findall(t.lower())


@click.command()
@click.option("--n", default=300, help="Figures to sample.")
@click.option("--model", default="qwen2.5vl:7b")
@click.option("--describe/--no-describe", default=True,
              help="Generate VLM descriptions (slow). --no-describe reuses cache.")
@click.option("--source", type=click.Choice(["local", "scicap"]), default="local",
              help="local = arxiv-rag's own 655 figures + the cached Phase 2 "
                   "descriptions (n=664, available now). scicap = outside "
                   "figures, but the image archive is 23.3 GB in 11 split parts.")
def main(n, model, describe, source):
    if source == "local":
        return _local(n)
    import base64
    import requests
    from tqdm import tqdm

    meta = json.loads((OUT / "val.json").read_text())
    imgs = {i["id"]: i for i in meta["images"]}
    anns = meta["annotations"]

    # Prefer Graph Plot: that is the figure kind arxiv-rag's rejection was about,
    # and mixing in tables/schematics would blur the comparison.
    rows = []
    for a in anns:
        im = imgs.get(a["image_id"])
        if not im or im.get("figure_type") != "Graph Plot":
            continue
        cap = (a.get("caption") or "").strip()
        if len(cap) < 60:
            continue
        p = OUT / "img" / im["file_name"]
        if not p.exists():
            continue
        rows.append({"id": a["image_id"], "caption": cap, "path": str(p)})
        if len(rows) >= n:
            break
    click.echo(f"{len(rows)} Graph Plot figures with captions and images")
    if not rows:
        raise SystemExit(f"no images found under {OUT/'img'} — unzip img-split first")

    cache_p = OUT / f"desc_{model.replace(':','_')}.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    if describe:
        todo = [r for r in rows if str(r["id"]) not in cache]
        for r in tqdm(todo, desc="describing"):
            b64 = base64.b64encode(open(r["path"], "rb").read()).decode()
            try:
                resp = requests.post(
                    "http://localhost:11434/api/chat",
                    json={"model": model, "stream": False,
                          "options": {"temperature": 0.0},
                          "messages": [{"role": "user", "images": [b64],
                                        "content": "Describe this scientific figure in 2-3 "
                                                   "sentences: what is plotted, the axes, and "
                                                   "the trend."}]},
                    timeout=180)
                resp.raise_for_status()
                cache[str(r["id"])] = resp.json()["message"]["content"].strip()
            except Exception as exc:                        # noqa: BLE001
                click.echo(f"\n  {r['id']}: {type(exc).__name__}")
                continue
            cache_p.write_text(json.dumps(cache, indent=2))
    rows = [r for r in rows if str(r["id"]) in cache]
    click.echo(f"{len(rows)} with descriptions")

    # ---- 1. DILUTION, measured directly on the text
    import collections
    df = collections.Counter()
    for r in rows:
        df.update(set(_toks(r["caption"])))
    n_docs = max(len(rows), 1)

    cap_share, rare_cap, rare_both, cap_len, desc_len = [], [], [], [], []
    for r in rows:
        c, d = _toks(r["caption"]), _toks(cache[str(r["id"])])
        cap_len.append(len(c))
        desc_len.append(len(d))
        cap_share.append(len(c) / max(len(c) + len(d), 1))
        # "rare" = appears in <5% of captions, i.e. the discriminative tokens
        rc = [t for t in c if df[t] / n_docs < 0.05]
        rare_cap.append(len(rc) / max(len(c), 1))
        rare_both.append(len(rc) / max(len(c) + len(d), 1))

    mean = lambda xs: sum(xs) / max(len(xs), 1)   # noqa: E731
    click.echo("\n=== 1. DILUTION (no queries needed) ===")
    click.echo(f"  caption length (tokens)        {mean(cap_len):>8.1f}")
    click.echo(f"  description length (tokens)    {mean(desc_len):>8.1f}")
    click.echo(f"  caption share of indexed text  {mean(cap_share):>8.1%}")
    click.echo(f"  rare-token density, caption    {mean(rare_cap):>8.1%}")
    click.echo(f"  rare-token density, cap+desc   {mean(rare_both):>8.1%}")
    drop = 1 - mean(rare_both) / max(mean(rare_cap), 1e-9)
    click.echo(f"  -> discriminative density falls {drop:>7.1%} when the "
               f"description is added")

    # ---- 2. KNOWN-ITEM RETRIEVAL, mechanical queries
    from rank_bm25 import BM25Okapi

    def known_item(texts):
        corpus = [_toks(t) for t in texts]
        bm = BM25Okapi(corpus)
        hits1 = hits5 = 0
        for i, r in enumerate(rows):
            q = _toks(r["caption"])
            q = q[:max(3, int(len(q) * 0.6))]          # mechanical truncation
            scores = bm.get_scores(q)
            order = sorted(range(len(corpus)), key=lambda j: -scores[j])
            if order[0] == i:
                hits1 += 1
            if i in order[:5]:
                hits5 += 1
        return hits1 / len(rows), hits5 / len(rows)

    cap_only = [r["caption"] for r in rows]
    cap_desc = [f"{r['caption']}\n{cache[str(r['id'])]}" for r in rows]
    a1, a5 = known_item(cap_only)
    b1, b5 = known_item(cap_desc)

    click.echo("\n=== 2. KNOWN-ITEM RETRIEVAL (query = 60% of caption) ===")
    click.echo(f"{'index':<22}{'recall@1':>10}{'recall@5':>10}")
    click.echo(f"{'caption only':<22}{a1:>10.3f}{a5:>10.3f}")
    click.echo(f"{'caption + VLM desc':<22}{b1:>10.3f}{b5:>10.3f}")
    click.echo(f"{'Δ':<22}{b1-a1:>+10.3f}{b5-a5:>+10.3f}")

    verdict = ("SUPPORTS the rejection: adding descriptions measurably hurts."
               if b5 < a5 - 0.005 else
               "DOES NOT support the rejection at this n: no measurable drop.")
    click.echo(f"\n  -> {verdict}")

    p = ROOT / "evals" / "scicap_dilution.json"
    p.write_text(json.dumps({
        "n": len(rows), "model": model,
        "caption_tokens": mean(cap_len), "desc_tokens": mean(desc_len),
        "caption_share": mean(cap_share),
        "rare_density_caption": mean(rare_cap),
        "rare_density_cap_desc": mean(rare_both),
        "known_item": {"caption_only": {"r@1": a1, "r@5": a5},
                       "caption_desc": {"r@1": b1, "r@5": b5}},
    }, indent=2) + "\n")
    click.echo(f"\nwrote {p}")


if __name__ == "__main__":
    main()
