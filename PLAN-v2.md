# arxiv-rag v2 — plan to make it resume-grade

Planning only. No code changes in this pass.

Baseline verified 2026-07-30 against the live index.

---

## Where it actually stands

| metric | value |
|---|---|
| corpus | 106 papers / 2834 chunks (115 PDFs on disk) |
| recall@5 | 93.33% (14/15), MRR 0.933 |
| by tag | acronym/ambiguous/distractor/easy/rare = 100%; **paraphrase = 83.3%** |
| negatives | 3 cases, mean top RRF 0.02707 |
| retrieval p50 | 8.2 ms e2e (embed 5.1 / dense 1.1 / bm25 1.8), ~121 QPS |
| generation | qwen2.5:14b via Ollama (Metal) |
| tests | 6 passing |

**Already strong and genuinely uncommon in a portfolio project:**

- Hybrid dense+sparse retrieval with RRF, and a written justification for why
  rank fusion beats score normalization.
- A tagged eval harness with per-tag breakdown — most side projects have none.
- Font-based section detection replacing regex, with the failure analysis that
  motivated it.
- Negative/OOD cases proving RRF scores are *not* a usable abstain signal.
- Measured hardware decisions (MPS 3.5x, batch-size asymmetry).

The problem is not that the work is weak. It's that **the strongest claims
aren't yet backed by the one table that would prove them.**

---

## The three credibility gaps

An interviewer who does retrieval for a living will find these in about
90 seconds. Each is cheap to close.

### Gap 1 — "hybrid is better" is asserted, never measured

The entire architectural thesis is that hybrid RRF beats either retriever
alone. There is no dense-only or BM25-only number anywhere in `EVAL.md`.
Right now the claim rests on one anecdote (`CLIP` scoring d=1/b=1).

Without the ablation the resume bullet is *"used hybrid retrieval."*
With it, the bullet is *"hybrid retrieval lifted recall@5 from X% to 93%."*

**This is the single highest-value item in this document.**

### Gap 2 — n=15 is not an eval set

15 scored cases, 6 of them one tag. One miss moves recall by 6.7 points, so
every number carries roughly ±7pp of noise. Worse, the labels were authored
against the original 20 papers and the corpus is now 106 — `EVAL.md` already
flags this as optimistic, which is honest but unresolved.

The self-flagged bias is *more* dangerous than the small n, because recall can
only look better as unlabeled relevant papers get added.

### Gap 3 — the efficiency story is unmeasured

There's a latency table for retrieval, but retrieval is 8 ms against seconds of
LLM generation. Optimizing it would be optimizing the wrong thing, and a sharp
interviewer will say so. **Ingest** is where the real engineering story is, and
it currently has no numbers at all.

---

## Phase A — Prove the design (do this first)

### A1. Ablation harness

Extend `eval_recall.py` with `--mode {dense,bm25,hybrid}` and emit one table:

| retriever | recall@5 | MRR | paraphrase | acronym | rare |
|---|---|---|---|---|---|
| dense only | ? | ? | ? | ? | ? |
| BM25 only | ? | ? | ? | ? | ? |
| hybrid RRF | 93.3% | 0.933 | 83.3% | 100% | 100% |

**Predicted, worth stating before running** — falsifiable predictions are the
point:

- `acronym` and `rare` should collapse under dense-only (POPE, VLUE, ELEVATER,
  Korean/German tokens are exactly what embeddings smear).
- `paraphrase` should collapse under BM25-only (no lexical overlap by design).
- Hybrid should beat both on the union.

If hybrid does *not* win, that is the most interesting possible result and
should be reported as-is. An eval that can only confirm you is not an eval.

Also sweep `_RRF_K ∈ {1, 10, 60, 200}` — one line in the table, demonstrates
the constant was chosen rather than copied.

**Effort:** ~1 hour. **Payoff:** the central resume claim becomes defensible.

### A2. Grow the eval set to 50+ cases, labeled against all 106 papers

Method that keeps this honest:

1. Sample ~40 held-out papers not represented in current labels.
2. For each, have qwen2.5:14b draft 2 candidate questions **from the abstract
   only** — never from retrieved chunks, or you leak the retriever into its own
   test set.
3. Hand-verify every one. Discard ambiguous items rather than guessing.
4. Keep the tag taxonomy; add `multi-hop` (needs ≥2 papers) and `temporal`
   ("what changed after 2024").
5. Re-audit the existing 15 against the full corpus — some now have additional
   valid relevant papers.

Target ≥50 scored + ≥10 negatives. Enough that one miss moves recall ~2pp.

**Effort:** 2–3 hours, mostly verification. **Payoff:** closes Gap 2, and
"I built and hand-validated a 50-case tagged eval set" is itself a bullet.

### A3. Report more than recall@5

Add recall@1, recall@10, nDCG@5. Recall@1 is what actually matters when the
top chunk dominates the generated answer, and it's a harder number to hit — so
it's more credible.

---

## Phase B — Close the quality gaps

### B1. Cross-encoder reranking

`cross-encoder/ms-marco-MiniLM-L-6-v2` over top-20, keep 5.

This is the standard next move in RAG and usually the largest single quality
win. Two reasons it fits here specifically:

- It should fix `paraphrase-hallucination`, the one persistent miss. A
  cross-encoder reads query and chunk *jointly*, so it can score
  "invent objects that are not in the image" against hallucination-benchmark
  text without needing shared vocabulary.
- **It adds latency, and measuring that tradeoff is the talking point.**
  Expect 8 ms → 50–100 ms. Report it as a tradeoff table, not a win.

Ship it behind `Config.rerank: bool = False` so the ablation can run both ways.

### B2. A real abstain signal

`EVAL.md` already proves the naive approach fails: negative queries scored
0.0164–0.0328, overlapping the positive range entirely. A flat RRF threshold is
unsafe. That's a good finding — now solve it.

Cross-encoder scores are calibrated in a way RRF ranks are not, so B1 makes
this possible: threshold on max rerank score, tuned on the negatives from A2.

Metric: false-abstain rate on positives vs. catch rate on negatives. Target
≥80% of negatives caught at <5% false abstains.

Interview value is high — "we found the obvious signal was unusable, and here's
what we replaced it with" is a much better story than never having checked.

### B3. Close the loop on the known miss

`paraphrase-hallucination` should either pass after B1 or get a written
explanation of why it can't. Either outcome is publishable; leaving it
unexplained is not.

---

## Phase C — Efficiency (the engineering story)

**First, the honest framing that makes this credible:** cold ingest is
dominated by arXiv's ToS-mandated 1.5 s inter-download delay — roughly 172 s of
the ~190 s for 115 papers, or ~90%. That is not optimizable without being rude
to a nonprofit, and pretending otherwise would be the wrong call.

So the target is **warm re-index** — re-chunking and re-indexing PDFs already on
disk. That's the actual developer loop (every parser or chunking change forces
one), and it's entirely CPU-bound.

### Measured baseline for warm re-index of 115 papers / 2834 chunks

| stage | cost | note |
|---|---|---|
| PDF parse | 13.9 s | 121 ms/pdf, **serial** |
| embedding | ~1.3 s | already MPS-batched |
| BM25 rebuild | ~5.6 s | **O(N²) across the run** |
| pickle writes | ~380 MB | 106 writes of a growing 7.2 MB file |
| **total** | **~21 s** | |

### C1. Kill the O(N²) BM25 rebuild

`add_chunks` calls `_rebuild_bm25()` + `_save_bm25()` on **every paper**.
`BM25Okapi` construction is linear (measured 0.037 ms/doc: 21 ms @ 500,
61 ms @ 1500, 106 ms @ 2834), so rebuilding once per paper over a growing
corpus is quadratic in total work.

At 106 papers: ~5.6 s and ~380 MB written. At 1000 papers it is ~560 s and
~38 GB — the thing that would actually stop this scaling.

Fix: a `flush()` / context-manager boundary so a batch rebuilds and persists
**once**. 5.6 s → ~0.1 s, and 380 MB → 7 MB.

Keeping per-call rebuild as the default for single adds is fine; the point is
the batch path shouldn't pay it 106 times.

### C2. Parallelize PDF parsing

Parsing is pure CPU, no shared state, 14 cores idle. `ProcessPoolExecutor`
over `parse_pdf` should take 13.9 s → ~1.5 s.

Caveat to verify, not assume: `Chunk` objects must pickle cleanly across the
process boundary, and PyMuPDF handles per-process init fine but shouldn't share
a `Document`.

### C3. Target

| | before | after |
|---|---|---|
| warm re-index (115 papers) | ~21 s | **~3 s** |
| disk written | ~380 MB | ~7 MB |

~7x, with the caveat that cold ingest stays network-bound — which is itself
the more impressive thing to say, because it shows you profiled before
optimizing.

### C4. Measure what the user feels

Retrieval is 8 ms; generation is seconds. Add end-to-end p50/p95 **including
generation**, and note that retrieval is <1% of it. Naming that explicitly is
better engineering judgment than shaving the 8 ms.

---

## Phase D — Packaging

### D1. README rewrite

Lead with the ablation table from A1. Current README explains what it does; it
should open by proving the design decision paid off. Include the architecture
diagram and the `demo.svg` already in `docs/`.

### D2. Draft resume bullets (fill the ?s from A1/C3)

> **arXiv-RAG** — hybrid retrieval over 106 ML papers (2.8k chunks). Ablation
> across a hand-labeled 50-case tagged eval set showed reciprocal-rank fusion
> lifted recall@5 from ?% (dense-only) to 93%, with cross-encoder reranking
> adding ?pp at ?ms p50.

> Profiled ingest and found 90% of wall-clock was ToS-mandated rate limiting;
> optimized the remaining CPU path instead — removed an O(N²) index rebuild and
> parallelized PDF parsing to cut warm re-index ~7x (21s → 3s).

> Replaced regex section detection with font-metric heuristics after error
> analysis showed table rows were being mis-split as headings, cutting
> unlabeled chunks from 3% to 1%.

The second bullet is the strongest of the three, because it demonstrates
profiling discipline and the judgment to *not* optimize the wrong thing —
rarer than raw speedup numbers.

---

## Sequence

| order | item | effort | why here |
|---|---|---|---|
| 1 | A1 ablation | 1 h | unblocks the main claim; cheapest high-value item |
| 2 | C1 BM25 batching | 1 h | makes every later re-index fast |
| 3 | C2 parallel parse | 1 h | same |
| 4 | A2 eval expansion | 3 h | slow, but everything downstream needs it |
| 5 | B1 rerank | 2 h | needs A2 to prove it helps |
| 6 | A3 + C4 metrics | 1 h | cheap once harness exists |
| 7 | B2 abstain | 2 h | needs B1 |
| 8 | D1/D2 packaging | 1 h | last |

A1 before C1 deliberately: the ablation is the resume-critical item, and doing
it first means even if the effort stops there, the project gained its most
important number.

---

## Explicit non-goals

- **Micro-optimizing the 8 ms retrieval path.** It's <1% of user-perceived
  latency. Doing it would signal poor prioritization.
- **Concurrent arXiv downloads.** Would violate ToS politeness for ~10% wall
  clock against a nonprofit. Not worth it, and saying why is the better answer.
- **Swapping to a bigger embedder** (`all-mpnet-base-v2`) before A1. Without
  the ablation there's no way to know if embeddings are even the bottleneck —
  the acronym/rare tags suggest BM25 is carrying that load.
- **Table extraction.** Real gap (flattened tables surface as hits), but it's a
  PDF-parsing project of its own and doesn't strengthen the retrieval story.

---

## Open items carried from NOTES-changes.md

- Stub `TODO:` docstrings still sit above completed code in `index.py`.
- `27 mess` false-positive heading, ~1.6% of chunks — deliberately unfixed.
- Models occasionally invent sibling names for stripped bibliography slots.
