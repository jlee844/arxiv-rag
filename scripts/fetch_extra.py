#!/usr/bin/env python3
"""Fetch the non-BEIR retrieval sets into the same plain-file layout.

RUN WITH .venv-beir.

LitSearch, BRIGHT and QASPER each ship a different structure, so each gets an
explicit adapter rather than a clever generic one. They are normalised to exactly
what eval_beir.py already reads -- corpus.jsonl / queries.jsonl / qrels.tsv --
so the SAME evaluator scores all six datasets. One scoring path is the point: a
per-dataset scorer would reintroduce the possibility of the metric differing
between runs, which is the failure this whole exercise exists to remove.

  LitSearch  597 expert-verified queries over 64k ML/NLP papers. Closest to the
             actual use case. qrels come from `corpusids` on each query.
  BRIGHT     reasoning-heavy; SOTA retrievers do badly by design. Per-domain, so
             --domain selects one. `excluded_ids` must be honoured or the number
             is wrong in the flattering direction.
  QASPER     1,585 arXiv NLP papers with evidence spans. Retrieval unit is a
             PARAGRAPH, not a paper, which is the one dataset here that actually
             exercises chunking.

    .venv-beir/bin/python scripts/fetch_extra.py --dataset litsearch
    .venv-beir/bin/python scripts/fetch_extra.py --dataset bright --domain biology
    .venv-beir/bin/python scripts/fetch_extra.py --dataset qasper
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import click

ROOT = Path(__file__).parent.parent


def _write(out: Path, corpus, queries, qrels):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w") as f:
        for cid, title, text in corpus:
            f.write(json.dumps({"_id": str(cid), "title": title or "",
                                "text": text or ""}) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for qid, text in queries:
            f.write(json.dumps({"_id": str(qid), "text": text}) + "\n")
    n = 0
    with (out / "qrels.tsv").open("w") as f:
        for qid, cid, s in qrels:
            f.write(f"{qid}\t{cid}\t{int(s)}\n")
            n += 1
    click.echo(f"  corpus  {len(corpus):>7}")
    click.echo(f"  queries {len(queries):>7}")
    click.echo(f"  qrels   {n:>7}")
    click.echo(f"  -> {out}")


def _aslist(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return [v]


@click.command()
@click.option("--dataset", type=click.Choice(["litsearch", "bright", "qasper"]),
              required=True)
@click.option("--domain", default="biology", help="BRIGHT domain.")
@click.option("--max-papers", default=400, help="QASPER papers to use.")
@click.option("--qasper-json", default="qasper-test-v0.3.json",
              help="Path to the official AllenAI QASPER json.")
def main(dataset, domain, max_papers, qasper_json):
    from datasets import load_dataset

    if dataset == "litsearch":
        q = load_dataset("princeton-nlp/LitSearch", "query", split="full")
        c = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
        corpus = [(r["corpusid"], r.get("title", ""), r.get("abstract", ""))
                  for r in c]
        queries, qrels = [], []
        for i, r in enumerate(q):
            qid = f"q{i}"
            queries.append((qid, r["query"]))
            for cid in _aslist(r["corpusids"]):
                qrels.append((qid, cid, 1))
        click.echo("litsearch:")
        _write(ROOT / "data" / "beir" / "litsearch", corpus, queries, qrels)

    elif dataset == "bright":
        ex = load_dataset("xlangai/BRIGHT", "examples", split=domain)
        dc = load_dataset("xlangai/BRIGHT", "documents", split=domain)
        corpus = [(r["id"], "", r["content"]) for r in dc]
        queries, qrels = [], []
        for r in ex:
            qid = str(r["id"])
            queries.append((qid, r["query"]))
            for cid in _aslist(r["gold_ids"]):
                qrels.append((qid, cid, 1))
        # excluded_ids are documents that must NOT be scored for that query
        # (near-duplicates of the answer). Dropping this rule inflates the score,
        # so it is written out for the evaluator to honour.
        excl = {str(r["id"]): [x for x in _aslist(r["excluded_ids"])
                               if x and x != "N/A"] for r in ex}
        out = ROOT / "data" / "beir" / f"bright-{domain}"
        click.echo(f"bright/{domain}:")
        _write(out, corpus, queries, qrels)
        (out / "excluded.json").write_text(json.dumps(excl, indent=2) + "\n")
        click.echo(f"  excluded ids written for "
                   f"{sum(1 for v in excl.values() if v)} queries")

    else:  # qasper
        # QASPER ships as a `datasets` loading script, and modern `datasets`
        # refuses to execute those at all -- trust_remote_code no longer helps.
        # The HF mirrors are question-level (question/answer/evidence) with NO
        # paragraph corpus, so they cannot support retrieval at all.
        #
        # So: the official AllenAI release, read as plain JSON. Download it with
        #   curl -sLO https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
        #   tar xzf qasper-test-and-evaluator-v0.3.tgz
        # and pass the extracted json via --qasper-json.
        src = Path(qasper_json).expanduser()
        if not src.exists():
            raise SystemExit(f"{src} not found; see the comment in this file "
                             f"for the download command")
        papers = json.loads(src.read_text())

        corpus, queries, qrels = [], [], []
        for pi, (pid, paper) in enumerate(papers.items()):
            if pi >= max_papers:
                break
            para_ids = []
            for si, sec in enumerate(paper.get("full_text", []) or []):
                name = sec.get("section_name") or ""
                for qi_, para in enumerate(sec.get("paragraphs", []) or []):
                    if not para or len(para) < 40:
                        continue
                    cid = f"{pid}-s{si}-p{qi_}"
                    corpus.append((cid, name, para))
                    para_ids.append((cid, para))

            for qi_, qa in enumerate(paper.get("qas", []) or []):
                qid = f"{pid}-q{qi_}"
                ev = []
                for a in (qa.get("answers", []) or []):
                    ev += [e for e in ((a.get("answer") or {}).get("evidence") or []) if e]
                if not ev:
                    continue
                matched = False
                for cid, para in para_ids:
                    # Prefix match: evidence strings are quoted verbatim from the
                    # paragraph, but may be truncated or re-joined across lines.
                    if any(e[:120] and e[:120] in para for e in ev):
                        qrels.append((qid, cid, 1))
                        matched = True
                if matched:
                    queries.append((qid, qa.get("question", "")))

        click.echo("qasper:")
        _write(ROOT / "data" / "beir" / "qasper", corpus, queries, qrels)


if __name__ == "__main__":
    main()
