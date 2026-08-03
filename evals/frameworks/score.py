"""Scoring shared by all three implementations. Deliberately dependency-free.

WHY THIS IS DUPLICATED FROM scripts/eval_recall.py RATHER THAN IMPORTED:

The framework implementations run in `.venv-frameworks`, which does not have
chromadb or the rest of this repo's pinned stack — importing `arxiv_rag.*`
would drag in the whole dependency tree and defeat the isolation that keeps a
LangChain version bump from touching the shipped app.

So the metric is re-implemented here in ~40 lines of stdlib, and
`test_frameworks_score.py` asserts it reproduces `eval_recall.py`'s hybrid
numbers exactly (96.91% / 0.936) when fed this repo's own retriever output.
A duplicated metric that is *pinned by a test* is safer than a shared metric
that forces a shared dependency tree.

TWO METRICS, because the repo turned out to have two and did not know it.

`eval_recall.py::_first_relevant_rank` returns the **chunk** index of the first
relevant hit — while its own docstring says "paper-level: scan unique paper
order", and the paper-level loop underneath it is unreachable dead code. So the
published MRR 0.936 is CHUNK-level. Meanwhile distill-lab's harness dedups to
papers first and computes 0.9381 on the same corpus and cases.

Both are defensible:
  - chunk-level  is OPERATIONAL: the generator sees 5 excerpts, so "rank" is
    the position among the excerpts actually shown. Stricter.
  - paper-level  is RETRIEVAL-THEORETIC: it asks whether the right document was
    found, independent of how many chunks of a neighbour preceded it.

Since paper_rank <= chunk_rank always, chunk-level is the pessimistic one.
The 0.0021 aggregate gap here is one `capability` case at chunk-rank 4 /
paper-rank 3, plus one `rare` case.

This module reports BOTH and gates on `mrr` (chunk-level), so the framework
arms stay comparable to the repo's published headline. Mixing the two across a
comparison table is the actual hazard, and reporting both is what prevents it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CASES = Path(__file__).parent.parent / "retrieval_cases.json"


def base_id(arxiv_id: str) -> str:
    """Strip the version suffix: '2306.09265v2' -> '2306.09265'."""
    return arxiv_id.split("v")[0]


def load_cases() -> tuple[list[dict], list[dict]]:
    """Returns (positives, negatives). Negatives have no `relevant` list."""
    data = json.loads(CASES.read_text())
    pos = [c for c in data if c.get("relevant")]
    neg = [c for c in data if not c.get("relevant")]
    return pos, neg


def chunk_rank(hits: list[dict], want: set[str]) -> int | None:
    """1-based CHUNK index of the first relevant hit. Matches eval_recall.py."""
    for i, h in enumerate(hits, 1):
        if base_id(h["arxiv_id"]) in want:
            return i
    return None


def paper_rank(hits: list[dict], want: set[str]) -> int | None:
    """1-based PAPER index of the first relevant hit, deduped in rank order.

    Always <= chunk_rank for the same hit list, since dedup can only remove
    positions ahead of the match, never add them.
    """
    papers: list[str] = []
    for h in hits:
        bid = base_id(h["arxiv_id"])
        if bid not in papers:
            papers.append(bid)
    for i, bid in enumerate(papers, 1):
        if bid in want:
            return i
    return None


def score(cases: list[dict], run, k: int = 5) -> dict:
    """Score a retriever over `cases`.

    Args:
        cases: positive eval cases (each with `query`, `relevant`, `tag`).
        run: callable(query) -> list of hit dicts with an `arxiv_id` key,
            already truncated to final_k by the implementation under test.
        k: recall@k.
    """
    hits_at_k = 0
    rr_sum = 0.0
    rr_paper_sum = 0.0
    by_tag: dict[str, dict] = defaultdict(lambda: {"n": 0, "hits": 0, "rr": 0.0})
    per_case = []

    for case in cases:
        want = {base_id(x) for x in case["relevant"]}
        results = run(case["query"])[:k]
        c_rank = chunk_rank(results, want)
        p_rank = paper_rank(results, want)
        rr = 1.0 / c_rank if c_rank else 0.0

        hits_at_k += bool(c_rank)
        rr_sum += rr
        rr_paper_sum += 1.0 / p_rank if p_rank else 0.0
        t = by_tag[case.get("tag", "untagged")]
        t["n"] += 1
        t["hits"] += bool(c_rank)
        t["rr"] += rr
        per_case.append({"id": case.get("id"), "tag": case.get("tag"),
                         "chunk_rank": c_rank, "paper_rank": p_rank, "rr": rr})

    n = max(len(cases), 1)
    return {
        "n": len(cases),
        "recall": hits_at_k / n,
        "mrr": rr_sum / n,                    # chunk-level; the gated metric
        "mrr_paper": rr_paper_sum / n,        # paper-level; distill-lab's metric
        "by_tag": {t: {"n": d["n"], "recall": d["hits"] / d["n"],
                       "mrr": d["rr"] / d["n"]}
                   for t, d in by_tag.items()},
        "per_case": per_case,
    }


def report(name: str, res: dict, timings: dict | None = None) -> None:
    print(f"\n{'─' * 60}\n{name}   n={res['n']}")
    print(f"  recall@5 {res['recall']:.4f}   MRR {res['mrr']:.4f} (chunk) "
          f"/ {res['mrr_paper']:.4f} (paper)")
    if timings:
        for label, val in timings.items():
            print(f"  {label:<22}{val}")
    print(f"\n  {'tag':<14}{'n':>4}{'recall':>9}{'MRR':>9}")
    for tag, d in sorted(res["by_tag"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {tag:<14}{d['n']:>4}{d['recall']:>9.3f}{d['mrr']:>9.3f}")
