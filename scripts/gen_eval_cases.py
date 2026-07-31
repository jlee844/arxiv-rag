#!/usr/bin/env python3
"""Draft candidate eval cases from paper abstracts, for HUMAN verification.

Why abstracts only: if questions were drafted from retrieved chunks, the
retriever would be shaping its own test set and every number afterwards would
be circular. The abstract is the paper's own summary and is never what we
score against — we score whether retrieval finds the PAPER.

Output is explicitly candidates/, not evals/. Nothing here is ground truth
until a human has confirmed it. See --review.

Usage:
    .venv/bin/python scripts/gen_eval_cases.py --n 10          # draft
    .venv/bin/python scripts/gen_eval_cases.py --review        # audit drafts
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from arxiv_rag.config import Config
from arxiv_rag.index import PaperIndex
from arxiv_rag.retrieve import retrieve

ROOT = Path(__file__).parent.parent
CASES = ROOT / "evals" / "retrieval_cases.json"
CANDIDATES = ROOT / "evals" / "candidate_cases.json"
REVIEW = ROOT / "evals" / "REVIEW.md"

_PROMPT = """\
You write search queries for evaluating a research-paper retrieval engine.

Below is one paper's title and abstract. Write exactly TWO queries a researcher \
might type into a search box that this paper would answer.

Query 1 — tag "paraphrase": describe the paper's contribution in DIFFERENT \
words. Do NOT reuse distinctive words from the title. No acronyms.
Query 2 — tag "rare": use the paper's most specific technical handle — the \
benchmark name, method name, dataset name, or a rare technical term it \
introduces. Short and precise.

Rules for both:
- 6 to 20 words, phrased as a natural search query or question.
- Never write "this paper", "the authors", "the study".
- Must be answerable from the abstract shown.

Return ONLY a JSON array, no prose:
[{"tag": "paraphrase", "query": "..."}, {"tag": "rare", "query": "..."}]

Title: {title}

Abstract: {abstract}
"""


def _base(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def _papers_with_abstracts(index: PaperIndex) -> dict[str, dict]:
    """{base_arxiv_id: {title, abstract}} pulled from the Abstract chunks."""
    got = index._col.get(include=["metadatas", "documents"])
    out: dict[str, dict] = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        if " ".join(meta["section"].lower().split()) != "abstract":
            continue
        bid = _base(meta["arxiv_id"])
        # keep the longest abstract chunk if a paper has several
        if bid not in out or len(doc) > len(out[bid]["abstract"]):
            out[bid] = {"title": meta["title"], "abstract": doc}
    return out


def _ask_llm(title: str, abstract: str, model: str) -> list[dict]:
    import ollama
    prompt = _PROMPT.replace("{title}", title).replace("{abstract}", abstract[:2500])
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.4},
    )
    text = resp["message"]["content"]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [i for i in items
            if isinstance(i, dict) and i.get("query") and i.get("tag")]


@click.command()
@click.option("--n", default=10, help="How many unlabeled papers to draft from")
@click.option("--seed", default=0, help="Sampling seed — keeps runs reproducible")
@click.option("--review", is_flag=True, help="Audit existing drafts against retrieval")
@click.option("--model", default=None, help="Ollama model (default: Config)")
@click.option("--review-file", is_flag=True, help="Write evals/REVIEW.md for hand-editing")
@click.option("--promote", is_flag=True, help="Merge REVIEW.md decisions into retrieval_cases.json")
def main(n, seed, review, model, review_file, promote) -> None:
    cfg = Config()
    cfg.embed_device = "cpu"          # reproducible scoring
    index = PaperIndex(cfg)
    model = model or cfg.ollama_model


    if promote:
        if not REVIEW.exists():
            click.echo("No evals/REVIEW.md — run --review-file first.", err=True)
            sys.exit(1)
        text = REVIEW.read_text()
        cands = {c["id"]: c for c in json.loads(CANDIDATES.read_text())}
        kept, dropped = [], 0
        for block in text.split("### ")[1:]:
            head = block.splitlines()[0]
            m = re.match(r"\[(KEEP|DROP)\]\s+(\S+)", head.strip())
            if not m:
                continue
            decision, cid = m.group(1), m.group(2)
            if decision == "DROP" or cid not in cands:
                dropped += 1
                continue
            case = dict(cands[cid])
            add = re.search(r"\*\*Also relevant.*?:\*\*(.*)", block)
            if add:
                extra_ids = [x.strip() for x in
                             re.split(r"[,\s]+", add.group(1).strip()) if x.strip()]
                for e in extra_ids:
                    if e not in case["relevant"]:
                        case["relevant"].append(e)
            case["source"] = "llm-draft-human-verified"
            kept.append(case)

        existing = json.loads(CASES.read_text())
        have = {c.get("id") for c in existing}
        new = [c for c in kept if c["id"] not in have]
        CASES.write_text(json.dumps(existing + new, indent=2) + "\n")
        click.echo(f"promoted {len(new)} new cases (dropped {dropped}) -> {CASES}")
        click.echo(f"retrieval_cases.json now has {len(existing) + len(new)} cases")
        return

    if review or review_file:
        if not CANDIDATES.exists():
            click.echo("No candidate_cases.json yet — run without --review first.",
                       err=True)
            sys.exit(1)
        cands = json.loads(CANDIDATES.read_text())
        meta = _papers_with_abstracts(index)
        clean = ambiguous = missed = 0
        rows = []

        for c in cands:
            want = set(c["relevant"])
            hits = retrieve(c["query"], index, cfg)
            got: list[str] = []
            for h in hits:
                b = _base(h["arxiv_id"])
                if b not in got:
                    got.append(b)

            found = bool(want & set(got))
            # Rank of the source paper, 1-based (None if absent).
            rank = next((i for i, g in enumerate(got, 1) if g in want), None)
            others = [g for g in got if g not in want]

            if not found:
                verdict, missed = "MISS ", missed + 1
            elif rank == 1:
                # Source is top hit. Other topically-similar papers below it are
                # expected in a 106-paper corpus of near-neighbours, not a
                # labeling problem. Earlier version flagged these as AMBIG on
                # len(others) alone, which mislabeled 4 of 5 perfectly good cases.
                verdict, clean = "OK   ", clean + 1
            elif len(others) >= 3:
                verdict, ambiguous = "AMBIG", ambiguous + 1
            else:
                verdict, clean = "OK   ", clean + 1

            rows.append((verdict.strip(), rank, c, got))
            if review_file:
                continue
            click.echo(f"\n[{verdict}] rank={rank} ({c['tag']}) {c['query']}")
            click.echo(f"    want: {sorted(want)}")
            for g in got[:5]:
                mark = "*" if g in want else " "
                click.echo(f"     {mark} {g}  {meta.get(g, {}).get('title', '?')[:62]}")

        click.echo("\n" + "─" * 60)
        click.echo(f"clean={clean}  ambiguous={ambiguous}  missed={missed}  "
                   f"total={len(cands)}")
        click.echo(
            "\nAMBIG = 3+ other papers in top-5. Usually means the label is "
            "INCOMPLETE (several papers really are relevant), not that retrieval "
            "failed. Add the genuinely relevant ids to 'relevant' before "
            "promoting into retrieval_cases.json."
        )
        click.echo("MISS = retrieval never found the source paper. Either a "
                   "genuinely hard case worth keeping, or a bad question. Judge "
                   "by hand.")

        if review_file:
            _write_review(rows, meta, clean, ambiguous, missed)
            click.echo(f"\nWrote {REVIEW}")
        return

    # ── draft mode ────────────────────────────────────────────────────────
    existing = json.loads(CASES.read_text())
    labeled = {r for c in existing for r in (c.get("relevant") or [])}
    already = {c["relevant"][0] for c in
               (json.loads(CANDIDATES.read_text()) if CANDIDATES.exists() else [])}

    meta = _papers_with_abstracts(index)
    pool = sorted(set(meta) - labeled - already)
    random.Random(seed).shuffle(pool)
    pool = pool[:n]

    click.echo(f"{len(meta)} papers indexed · {len(labeled)} labeled · "
               f"drafting from {len(pool)} unlabeled\n")

    drafted = []
    for bid in pool:
        info = meta[bid]
        items = _ask_llm(info["title"], info["abstract"], model)
        if not items:
            click.echo(f"  [skip] {bid}: model returned no usable JSON")
            continue
        for it in items:
            drafted.append({
                "id": f"{it['tag']}-{bid}",
                "tag": it["tag"],
                "query": it["query"].strip(),
                "relevant": [bid],
                "source": "llm-draft-unverified",
            })
        click.echo(f"  {bid}  {info['title'][:56]}")
        for it in items:
            click.echo(f"      ({it['tag']}) {it['query']}")

    prior = json.loads(CANDIDATES.read_text()) if CANDIDATES.exists() else []
    CANDIDATES.write_text(json.dumps(prior + drafted, indent=2) + "\n")
    click.echo(f"\nWrote {len(drafted)} candidates "
               f"({len(prior) + len(drafted)} total) -> {CANDIDATES}")
    click.echo("NOT ground truth. Run --review, verify by hand, then promote.")


_ORDER = {"MISS": 0, "AMBIG": 1, "OK": 2}


def _write_review(rows, meta, clean, ambiguous, missed) -> None:
    """Emit a hand-editable review file. Flagged cases first."""
    rows = sorted(rows, key=lambda r: (_ORDER.get(r[0], 9), r[2]["tag"]))
    out = [
        "# Eval case review",
        "",
        f"{clean} clean · {ambiguous} ambiguous · {missed} missed · "
        f"{len(rows)} total candidates.",
        "",
        "## How to use",
        "",
        "Edit the `[KEEP]` / `[DROP]` marker on each `###` heading. Everything",
        "defaults to KEEP. When done:",
        "",
        "```bash",
        ".venv/bin/python scripts/gen_eval_cases.py --promote",
        "```",
        "",
        "**Judge a MISS by reading the question and paper — NOT by whether",
        "retrieval found it.** Dropping cases because the current system fails",
        "them strips out exactly what the eval exists to catch, and quietly",
        "inflates recall. Some misses SHOULD survive.",
        "",
        "If other retrieved papers are genuinely relevant too, add their ids to",
        "the `Also relevant` line — an incomplete label reads as a false miss.",
        "",
        "---",
        "",
    ]
    for verdict, rank, c, got in rows:
        src = c["relevant"][0]
        out += [
            f"### [KEEP] {c['id']}",
            "",
            f"- **verdict:** `{verdict}`  ·  **source rank:** `{rank}`  ·  "
            f"**tag:** `{c['tag']}`",
            f"- **query:** {c['query']}",
            f"- **labeled:** `{src}` — {meta.get(src, {}).get('title', '?')}",
            "- **retrieved top-5:**",
        ]
        for i, g in enumerate(got[:5], 1):
            mark = "**<-- labeled**" if g == src else ""
            out.append(f"  {i}. `{g}` {meta.get(g, {}).get('title', '?')[:70]} {mark}")
        out += ["- **Also relevant (add ids, space-separated):** ", "", "---", ""]
    REVIEW.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
