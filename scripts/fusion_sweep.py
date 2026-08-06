#!/usr/bin/env python3
"""Can weighted fusion fix the two datasets where hybrid loses?

THE FINDING THIS INTERROGATES: across six datasets, hybrid beat both single
arms exactly when BM25 >= dense, and lost exactly when dense > BM25 -- 6/6, no
exceptions. In both losses hybrid landed BETWEEN the two arms, which is what
equal-weight RRF does: it averages rankings, so it drags the stronger retriever
toward the weaker one.

    scifact   bm25 .686 dense .645 -> hybrid .712   bm25 stronger, fusion helps
    nfcorpus  bm25 .324 dense .316 -> hybrid .349   bm25 stronger, fusion helps
    qasper    bm25 .142 dense .105 -> hybrid .146   bm25 stronger, fusion helps
    litsearch bm25 .413 dense .354 -> hybrid .461   bm25 stronger, fusion helps
    scidocs   bm25 .157 dense .216 -> hybrid .203   dense stronger, fusion HURTS
    bright    bm25 .075 dense .137 -> hybrid .112   dense stronger, fusion HURTS

So: does *weighting* the arms recover the loss, and is there a weight that is
good everywhere?

RETRIEVE ONCE, FUSE MANY TIMES. The rankings per arm are fixed; only the fusion
is swept. Re-retrieving per weight would take hours and produce identical
inputs.

THE HONEST TRAP, STATED UP FRONT: picking the best weight per dataset *using the
test labels* is not a method, it is an oracle. It measures the HEADROOM available
to any adaptive scheme, nothing more. It is reported as `oracle`, clearly
labelled, and must never be quoted as a system result. A real fix needs a weight
chosen from a label-free signal -- which is what `--probe` tests.

    .venv/bin/python scripts/fusion_sweep.py --dataset scidocs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from tqdm import tqdm

ROOT = Path(__file__).parent.parent


def _load(dataset):
    d = ROOT / "data" / "beir" / dataset
    corpus = [json.loads(l) for l in (d / "corpus.jsonl").read_text().splitlines()]
    queries = {json.loads(l)["_id"]: json.loads(l)["text"]
               for l in (d / "queries.jsonl").read_text().splitlines()}
    qrels = {}
    for line in (d / "qrels.tsv").read_text().splitlines():
        q, c, s = line.split("\t")
        qrels.setdefault(q, {})[c] = int(s)
    return corpus, queries, qrels


def _wrrf(bm25_ids, dense_ids, w_bm25, k=60, depth=100):
    """Weighted reciprocal-rank fusion. w_bm25=1 -> BM25 only, 0 -> dense only."""
    scores = {}
    for r, did in enumerate(bm25_ids, 1):
        scores[did] = scores.get(did, 0.0) + w_bm25 / (k + r)
    for r, did in enumerate(dense_ids, 1):
        scores[did] = scores.get(did, 0.0) + (1.0 - w_bm25) / (k + r)
    return sorted(scores, key=scores.get, reverse=True)[:depth]


@click.command()
@click.option("--dataset", required=True)
@click.option("--depth", default=100)
@click.option("--rrf-k", default=60)
def main(dataset, depth, rrf_k):
    import pytrec_eval

    from arxiv_rag.config import Config
    from arxiv_rag.index import PaperIndex
    from arxiv_rag.retrieve import retrieve_bm25, retrieve_dense

    corpus, queries, qrels = _load(dataset)
    qids = [q for q in qrels if q in queries]

    idx_dir = ROOT / "data" / "beir" / dataset / "index"
    if not (idx_dir / "chroma").exists():
        raise SystemExit(f"no cached index for {dataset}; run eval_beir.py first")
    cfg = Config()
    cfg.chroma_dir, cfg.bm25_dir = idx_dir / "chroma", idx_dir / "bm25"
    cfg.top_k = cfg.final_k = depth
    cfg.rerank = False
    cfg.query_expansion = None
    index = PaperIndex(cfg)
    click.echo(f"{dataset}: {index.count()} docs · {len(qids)} queries (cached index)")

    # Retrieve each arm ONCE.
    runs = {}
    for qid in tqdm(qids, desc="retrieving both arms"):
        q = queries[qid]
        runs[qid] = ([h["arxiv_id"] for h in retrieve_bm25(q, index, cfg)][:depth],
                     [h["arxiv_id"] for h in retrieve_dense(q, index, cfg)][:depth])

    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in qids},
                                        {"ndcg_cut.10"})

    def score(w):
        run = {}
        for qid, (b, d) in runs.items():
            fused = _wrrf(b, d, w, k=rrf_k, depth=depth)
            run[qid] = {did: float(len(fused) - i) for i, did in enumerate(fused)}
        s = ev.evaluate(run)
        return sum(x["ndcg_cut_10"] for x in s.values()) / max(len(s), 1)

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    click.echo(f"\n{'w_bm25':>8}{'nDCG@10':>10}   (0 = dense only, 1 = BM25 only)")
    click.echo("-" * 34)
    out = {}
    for w in weights:
        v = score(w)
        out[w] = v
        mark = "  <- shipped (equal weight)" if abs(w - 0.5) < 1e-9 else ""
        click.echo(f"{w:>8.1f}{v:>10.4f}{mark}")

    best_w = max(out, key=out.get)
    click.echo(f"\nbest weight {best_w:.1f} -> {out[best_w]:.4f}"
               f"   (equal-weight 0.5 -> {out[0.5]:.4f}, "
               f"delta {out[best_w]-out[0.5]:+.4f})")
    click.echo(f"dense-only {out[0.0]:.4f} · BM25-only {out[1.0]:.4f}")
    # Threshold, not 1e-6. The first version declared "a weighted blend BEATS
    # both single arms" on SCIDOCS off a margin of +0.0001 -- 0.2165 vs 0.2164,
    # which is noise dressed as a result. Require a margin worth acting on.
    MEANINGFUL = 0.005
    best_single = max(out[0.0], out[1.0])
    gain = out[best_w] - best_single
    if gain > MEANINGFUL:
        click.echo(f"  -> a weighted blend beats the best single arm by "
                   f"{gain:+.4f}. Fusion is not the problem; equal weighting was.")
    else:
        click.echo(f"  -> NO blend meaningfully beats the best single arm "
                   f"(best gain {gain:+.4f} < {MEANINGFUL}). On this corpus the "
                   f"right move is to PICK the arm, not tune the mix.")
    click.echo("\nNOTE: `best weight` is chosen USING THE TEST LABELS. It is an "
               "oracle\n      upper bound on any adaptive scheme, not a system "
               "result.")

    p = ROOT / "evals" / f"fusion_{dataset}.json"
    p.write_text(json.dumps({"dataset": dataset, "rrf_k": rrf_k,
                             "ndcg_by_weight": out, "oracle_w": best_w},
                            indent=2) + "\n")
    click.echo(f"\nwrote {p}")


if __name__ == "__main__":
    main()
