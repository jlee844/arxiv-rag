# Evaluation

Measured on Apple Silicon (M4 Max, 36 GB). Corpus: **116 papers / 3801 chunks**
(text + figure captions). The header previously read 109/2960, which was stale
by two corpus expansions.

Cases: `evals/retrieval_cases.json` · runner: `scripts/eval_recall.py` ·
latency: `scripts/bench_latency.py`

```bash
.venv/bin/python scripts/eval_recall.py --ablate     # retriever comparison
.venv/bin/python scripts/eval_recall.py --gate       # abstain threshold sweep
.venv/bin/python scripts/eval_recall.py --rerank     # cross-encoder (rejected)
.venv/bin/python scripts/bench_latency.py --rounds 15
```

## Eval set

**119 cases = 97 positives + 22 negatives.**

| source | n | provenance |
|---|---|---|
| original | 18 | hand-written |
| `hand-written-negative` | 19 | hand-written adversarial negatives |
| `llm-draft-auto-triaged` | 61 | LLM-drafted from abstracts, heuristically triaged |
| thin-tag expansion | 13 | hand-written, verified against retrieval before adding |
| capability slice | 8 | hand-written over 6 newly ingested capability benchmarks |

Two anti-circularity rules the set is built on:

1. **Questions are drafted from abstracts only**, never from retrieved chunks —
   otherwise the retriever shapes its own test set.
2. **A failing case is judged by reading the question and paper**, never by
   whether retrieval found it. 61 of 62 candidates were promoted regardless of
   verdict; the one dropped had a genuinely wrong label. Dropping cases the
   system fails would strip out exactly what the eval exists to catch.

⚠️ The 61 auto-triaged cases were **not read individually**. Numbers below
should be read as "auto-generated and triaged", not "hand-validated".

## Standard benchmarks (BEIR + LitSearch + QASPER + BRIGHT)

Everything else in this file is measured on 145 cases that I wrote and then
tuned the retriever against. Honestly computed, and **not defensible to a
skeptic**: cases written by the tuner encode assumptions the retriever already
satisfies, which is how a metric reaches 98.97%.

This section is the answer to "who wrote your eval?" — someone else's corpora,
someone else's queries, someone else's metric, scored by `pytrec_eval` rather
than by code of mine.

| dataset | docs | queries | BM25 | dense | hybrid | published BM25 | best |
|---|---|---|---|---|---|---|---|
| SciFact | 5,183 | 300 | 0.6857 | 0.6451 | **0.7120** | 0.665 | hybrid |
| NFCorpus | 3,633 | 323 | 0.3235 | 0.3164 | **0.3493** | 0.325 | hybrid |
| SCIDOCS | 25,657 | 1,000 | 0.1572 | **0.2164** | 0.2033 | 0.158 | dense |
| QASPER | 18,173 | 1,260 | 0.1415 | 0.1047 | **0.1460** | — | hybrid |
| LitSearch | 64,183 | 597 | 0.4125 | 0.3540 | **0.4606** | — | hybrid |
| BRIGHT-bio | 57,359 | 103 | 0.0747 | **0.1365** | 0.1122 | — | dense |

nDCG@10, retrieval depth 100, reranking off (shipped default).

### The harness reproduces published baselines

Our BM25 lands at 0.6857 / 0.3235 / 0.1572 against the BEIR paper's 0.665 /
0.325 / 0.158. Matching a published number on three independent datasets is the
strongest available evidence that the scoring is right — and it is a check the
hand-built eval structurally cannot perform, because there is nothing to
reproduce.

`scripts/eval_beir.py` treats a BM25 score far ABOVE published as a **bug
signal** (relevance leakage, id mismatch), not a win.

### The hybrid claim was too strong, and the correction has a mechanism

Hybrid wins **exactly** when BM25 >= dense, and loses **exactly** when dense >
BM25. Six datasets, no exceptions. In both losses hybrid lands *between* the two
arms — SCIDOCS 0.2033 between 0.157 and 0.216; BRIGHT 0.1122 between 0.075 and
0.137 — which is what equal-weight RRF does: it averages rankings, so it drags
the stronger retriever toward the weaker one.

**This confirms a diagnosis already in this file at n=6.** The `capability`
slice above is the one place the in-house eval saw fusion lose to dense (0.867
vs 0.889), and the stated cause was "BM25 has no lexical signal to find... RRF
then lets that noise pull hybrid below dense." That was 6 cases and could not be
resolved from noise. BEIR reproduces the same mechanism at 1,000+ queries on
whole corpora where BM25 is weak.

So the honest claim is not "hybrid RRF is better". It is:

> **Equal-weight RRF helps only when the lexical arm is competitive with the
> dense arm. Where BM25 collapses, fusion costs quality.**

### Weighted fusion does NOT rescue it

SCIDOCS, sweeping the BM25 weight (`scripts/fusion_sweep.py`):

| w_bm25 | 0.0 | 0.1 | 0.3 | **0.5 (shipped)** | 0.7 | 1.0 |
|---|---|---|---|---|---|---|
| nDCG@10 | 0.2164 | 0.2165 | 0.2102 | **0.2026** | 0.1905 | 0.1572 |

Monotonic decline in BM25 weight. The best blend beats dense-only by **+0.0001**,
which is noise. **No weighting recovers the loss** — on a corpus where BM25 is
weak the right move is to pick the arm, not tune the mix.

(The first version of that script declared "a weighted blend BEATS both single
arms" off that +0.0001, because its threshold was 1e-6. Fixed to 0.005. Noise
dressed as a result is the exact failure this section exists to remove.)

Swept across all six datasets:

| dataset | dense | BM25 | equal 0.5 | **w=0.7** | argmax w | gain vs best single arm |
|---|---|---|---|---|---|---|
| SciFact | 0.6451 | 0.6857 | 0.7141 | **0.7179** | 0.7 | **+0.0322** |
| LitSearch | 0.3540 | 0.4125 | 0.4629 | **0.4677** | 0.7 | **+0.0552** |
| NFCorpus | 0.3164 | 0.3311 | 0.3519 | **0.3524** | 0.6 | **+0.0225** |
| QASPER | 0.1047 | 0.1415 | 0.1469 | **0.1514** | 0.7 | **+0.0099** |
| SCIDOCS | 0.2164 | 0.1572 | 0.2026 | 0.1905 | 0.1 | +0.0001 |
| BRIGHT-bio | 0.1365 | 0.0747 | 0.1118 | 0.0993 | 0.0 | +0.0000 |

Two separate conclusions, one an order of magnitude bigger than the other:

1. **Where BM25 collapses, no weight helps — and the fix is large.** SCIDOCS and
   BRIGHT decline monotonically in BM25 weight; the optimum is pure dense, worth
   **+0.014 and +0.025** over equal weighting. Arm *selection*, not mixing.
2. **Where fusion works, w=0.7 beats equal weighting on 4 of 4**, by +0.0038,
   +0.0048, +0.0005, +0.0045 (mean +0.0034). Small, consistent, free.

> A note on how conclusion 2 was nearly missed. Comparing ARGMAX weights gives
> 0.7 / 0.7 / 0.6 / 0.7, which reads as "no single constant" — and that was the
> first reading. But the right question is not where each curve peaks, it is
> whether one fixed weight beats the shipped one everywhere. NFCorpus's argmax of
> 0.6 puts w=0.7 just 0.0012 below its own optimum, which is noise. Argmax
> jitter is not disagreement.

**Still not adopted.** Every weight here was scored against test labels, so
+0.0034 is an oracle-flavoured gain on six datasets, not a validated default.
The honest status: a defensible candidate for `Config.rrf`, worth far less than
conclusion 1, and it should be confirmed on a held-out dataset before shipping.

### Honest limits of this section

- **These corpora are text-only.** The figure, table and multimodal work cannot
  be validated this way and remains measured on the purpose-built corpus. The
  claim is "retrieval quality is validated on standard benchmarks; multimodal
  and safety work is validated in-house", not that everything moved.
- **QASPER, LitSearch and BRIGHT have no published BM25 here**, so they get no
  reproduction guarantee — only the three BEIR sets do.
- **BRIGHT is one domain** (biology) of eleven.
- **Absolute numbers look worse than the in-house ones and that is the point.**
  0.712 nDCG@10 on SciFact is a more credible statement than 98.97% recall@5 on
  cases I wrote.

## Retriever ablation

Measured at n=97 positives. Every number in the *rejected-techniques* table
further down was measured at **n=76** and has NOT been re-run against the larger
set — those rows are labelled accordingly rather than silently rescaled.

| retriever | recall@5 | MRR | paraphrase (37) | rare (32) | capability (6) |
|---|---|---|---|---|---|
| dense only | 94.85% | 0.888 | 89% | 97% | MRR **0.889** |
| BM25 only | 91.75% | 0.862 | 81% | 100% | MRR 0.611 |
| **hybrid RRF** | **97.94%** | **0.943** | **95%** | **100%** | MRR 0.867 |

> **Tokenizer change, 2026-08-03.** These are post-stemming numbers. The BM25
> tokenizer was `text.lower().split()` until the E1 framework ablation showed
> LlamaIndex's stemmed tokenizer accounted for its entire retrieval lead; see
> [`evals/frameworks/README.md`](evals/frameworks/README.md). Hybrid moved
> 96.91% / 0.936 → 97.94% / 0.943, BM25-only 0.840 → 0.862, dense unchanged.
> **Every row in the rejected-techniques table below predates this change** and
> was measured against the old tokenizer — they are not re-run here, and are
> labelled rather than silently rescaled.

**The `capability` slice is the first place fusion is measurably WORSE than a
single retriever** — dense 0.889 vs hybrid 0.867. These queries name a *capability*
("physical properties like mass and friction", "which region of an object a person
could act on") while every candidate paper is topically identical ("a benchmark for
VLMs"), so BM25 has nothing to lock onto: MRR 0.611, its worst slice by a wide
margin. RRF then lets that noise pull hybrid below dense. Small n (6) — but it is
the sharpest statement of the dense/BM25 split in the whole eval, and a concrete
target for a reranker.

Stemming did not rescue this slice and slightly widened the gap (hybrid 0.875 →
0.867 even as BM25-only rose 0.556 → 0.611), which is consistent with the
diagnosis: the problem is that BM25 has *no lexical signal to find*, not that it
was matching the signal badly.

**The 13 thin-tag cases raised every number, which is a warning, not a win.**
12 of 13 land at rank 1 for hybrid, so they add resolution to `acronym` (3→7),
`distractor` (1→5), `ambiguous` (2→5) and `easy` (1→3) — the tags that were
previously too small to report — while *reducing* headroom. Only
`acronym-minigpt4` (hybrid rank 2) is unsolved. A benchmark whose cases are
already answered cannot measure a reranker; the next cases added must be
harder, not merely more numerous.

Hybrid beats the best single retriever by **+2.25pp recall and +0.053 MRR**,
and the per-tag split shows exactly why: dense wins paraphrase (89 vs 81), BM25
wins rare technical tokens (100 vs 97), fusion takes both.

**This conclusion reversed when the eval set grew.** At n=15 all three tied at
93.33% recall and fusion looked like a ranking-only win. The tie was an artifact
of insufficient resolution — at n=15 a single case moves recall 6.7pp, larger
than the effect being measured.

**Known miss:** `paraphrase-hallucination` — *"datasets that check whether VLMs
invent objects that are not in the image"* is missed by all three retrievers and
by the rejected cross-encoder.

## Abstention / out-of-distribution

RRF scores **cannot** gate: they fuse ranks and discard magnitude by
construction, which is precisely the signal a relevance gate needs. Raw dense
cosine retains it.

Threshold sweep over 76 positives / 22 negatives (n=76 — predates the
thin-tag expansion; not re-run):

| threshold | false-abstain | negatives caught |
|---|---|---|
| **0.35 (shipped)** | **0/76 (0%)** | **68.2%** |
| 0.37 | 1/76 (1.3%) | 72.7% |
| 0.40 | 1/76 (1.3%) | 81.8% |
| 0.50 | 7/76 (9.2%) | 95.5% |

**Positives and negatives overlap** (max negative 0.6231 > min positive 0.3572),
so no threshold separates them cleanly.

Residual leaks are off-topic questions in on-topic vocabulary:

```
0.6231  what learning rate should I use with the Adam optimizer
0.4598  how do I fix a CUDA out of memory error during training
```

The lowest-scoring *positives* are all short rare-token queries:

```
0.3572  THaMES framework
0.4129  AI2-THOR environment with SAM and PPO
0.4408  causal tracing tool for BLIP activations
```

**The gate reads dense cosine only, so it inherits dense's blind spot** — it is
weakest exactly where BM25 carries retrieval. Chose 0/76 false-abstain over 82%
catch: refusing a user who typed a real paper name looks broken.

Two earlier published versions of this table were wrong: one claimed clean
separation (+0.0669) from too-easy negatives, and one was measured at n=15
positives and never re-run after the eval set grew. See `NOTES-changes.md`
§10/§15.

## Standard improvements tested and rejected

**All rows below were measured against the pre-stemming BM25 tokenizer**, i.e.
against a hybrid baseline of 96.91% / 0.936 rather than today's 97.94% / 0.943.
The verdicts are not automatically invalidated — but a technique that lost by a
hair to the old baseline would lose by more to the new one, and the
cross-encoder's `rare`-slice win in particular is worth re-measuring now that
BM25 handles morphology itself.

| technique | verdict |
|---|---|
| cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`, top-20→5) | **Re-measured at n=97:** recall 96.91% → 94.85%, MRR 0.936 → 0.900, +82 ms. Still rejected as a default — but **not uniformly worse**, see the slice split below. Kept behind `Config.rerank=False`, reproduces via `--rerank`. **It scored the prompt-injection chunk highest of all 22 negatives** — a better relevance model is a better injection amplifier. |
| prompt hardening vs injection | byte-identical output; zero effect |
| larger embedder (`all-mpnet-base-v2`) | dense recall 93.42% → 90.79%, MRR 0.878 → 0.824, 3.5× slower/query, 37× slower to index |
| larger `top_k` (12/16/20/30) | recall moves ±1 case (noise); MRR declines monotonically |
| HyDE (hypothetical document embeddings) | recall 96.05% → 94.74%, 61× latency; invented specifics pull retrieval toward non-existent papers |
| **hybrid RRF over single retriever** | **kept** — +2.6pp recall, +0.06 MRR |
| **multi-query expansion** | **kept, opt-in** — recall 96.05% → **98.68%**, fixes all known misses; costs 23× latency and −0.031 MRR |

### Cross-encoder reranking, by slice (n=97)

The aggregate rejection hides an opposite result on two slices:

| tag | n | hybrid | + cross-encoder | Δ |
|---|---|---|---|---|
| **capability** | 6 | 0.875 | **0.917** | **+0.042** |
| **rare** | 32 | 0.975 | **1.000** | **+0.025** |
| distractor | 5 | 1.000 | 1.000 | 0 |
| easy | 3 | 1.000 | 1.000 | 0 |
| acronym | 7 | 0.929 | 0.857 | −0.072 |
| paraphrase | 37 | 0.901 | 0.827 | −0.074 |
| ambiguous | 7 | 0.900 | 0.750 | −0.150 |

The cross-encoder wins exactly where fusion is weakest — `capability`, the only
slice where hybrid scores below dense alone — and loses on `paraphrase` and
`ambiguous`, which together are 44 of 97 positives and therefore decide the
aggregate.

Mechanism: a cross-encoder judges whether a passage *answers* the query, which
is what `capability` queries need and what a bi-encoder cannot do. On
`ambiguous` it actively hurts, because those cases have several correct papers
and the reranker collapses toward one — recall drops 100% → 85.7%, the only
slice where reranking loses a document outright rather than just reordering.

**This refines the rejection rather than reversing it.** The shipped default is
unchanged. But "reranking does not work here" is too strong; the accurate claim
is that this cross-encoder trades paraphrase and multi-answer performance for
capability and rare-token performance, and on this corpus that trade is bad.

## Query expansion

| | recall@5 | MRR | abstain AUC | latency |
|---|---|---|---|---|
| hybrid (baseline) | 96.05% | **0.936** | 0.975 | **56 ms** |
| **+ multi-query** | **98.68%** | 0.905 | 0.975 | 1284 ms |
| + HyDE | 94.74% | 0.910 | 0.819 | 3442 ms |

Multi-query rewriting fixes `paraphrase-hallucination` — the case that survived
reranking, every `top_k`, and a larger embedder. The mechanism:

```
original : datasets that check whether VLMs invent objects that are not in the image
rewrite  : evaluating visual language models for object hallucination in images
```

The query avoids the field's term of art, so BM25 had zero lexical overlap and
contributed nothing. The rewrite supplies "hallucination" and BM25 finds it
immediately.

**Off by default** (`Config.query_expansion = None`): 23× retrieval latency is
defensible on `/api/chat`, where generation already dominates, but not on
`/api/search`, which exists to be fast.

The narrow claim: on a single-domain corpus where nearly every chunk is
topically adjacent to every query, hybrid RRF over a small fast embedder is
already a strong baseline and the usual upgrades buy nothing measurable.

### The one persistent miss, root-caused

`paraphrase-hallucination` is missed by every configuration above. It is **not**
a semantic failure — the target chunk ranks 11 of 2960 in the full dense
ordering (cosine 0.4929, top 0.4%). It loses because RRF at K=60 is deliberately
flat:

```
target (dense#11, dense-only)     rrf = 0.01408
any consensus hit (both, rank 1)  rrf = 0.03279    2.3x
```

That flatness is what makes consensus outweigh single-retriever confidence — and
the same property means a deep single-retriever hit can never climb. Lowering K
would widen the gap, not close it. Fixing this needs query expansion or a
domain-tuned embedder; neither is tested.

## Framework ablation (E1): hand-rolled vs LlamaIndex vs LangChain

Same 3288 chunks, same 97 cases, same embedder, same `top_k`/`final_k`/RRF k=60.
Only the implementation of hybrid retrieval varies. Full method and friction log:
[`evals/frameworks/README.md`](evals/frameworks/README.md).

| implementation | recall@5 | MRR | build | query p50 |
|---|---|---|---|---|
| hand-rolled, pre-stemming | 96.91% | 0.9359 | 2.8 s warm | 11.2 ms |
| LangChain `EnsembleRetriever` (FAISS + BM25) | 96.91% | 0.9261 | 6.1 s | 9.9 ms |
| LlamaIndex `QueryFusionRetriever` | **98.97%** | 0.9433 | 8.3 s | 33.5 ms |
| LlamaIndex, `skip_stemming=True` | 96.91% | 0.9287 | 10.4 s | 33.5 ms |
| **hand-rolled + stemming (shipped)** | **97.94%** | **0.9428** | 4.6 s warm | **9.5 ms** |

**The framework advantage was a tokenizer default, not an abstraction.**
LlamaIndex led by 2.06pp until `skip_stemming=True` dropped it to *exactly* the
hand-rolled score — 96.91% recall, 0.919 paraphrase recall, both to the digit.
LangChain's BM25 also defaults to `.split()` and matched the old baseline too.

```
query: "evaluating hallucinations in multimodal models"
  LlamaIndex : ['evalu', 'hallucin', 'multimod', 'model']
  arxiv-rag  : ['evaluating', 'hallucinations', 'in', 'multimodal', 'models']
```

Porting stemming + punctuation splitting + stopword removal into
`arxiv_rag/index.py::_tokenize` captured the gain and matched LlamaIndex's MRR
(0.9428 vs 0.9433) at 3.5× lower query latency. **Dense-only scores were
byte-identical before and after** (94.85% / 0.888), which is the control proving
only the sparse path moved.

Not free: `acronym` MRR fell 0.929 → 0.857 (one case of seven) and `capability`
0.875 → 0.867, paying for `paraphrase` recall 0.919 → 0.946, `rare` MRR
0.975 → 1.000, and `ambiguous` 0.929 → 1.000.

The honest conclusion is not "frameworks add nothing." It is that **the
frameworks were valuable as an oracle rather than as a dependency** — they
exposed a defect that a year of self-comparison could not, because this repo's
eval only ever compared this repo against itself.

## Multimodal: figure retrieval

664 figures extracted from 114/135 papers and indexed as caption chunks, then
measured on both sides. Reproduce with `scripts/eval_figures.py`.

**Extraction is clip-render, not image extraction.** `page.get_images()` finds
462 embedded bitmaps in a 25-paper sample and misses **73 pages of vector
figures** — matplotlib/TikZ plots are PDF drawing operators, not bitmaps, so the
scatter plots and ablation charts a reader most wants return nothing, silently.
Rendering the figure *region* captures vector and raster identically.

### The trade, measured on both sides

| | recall@5 | MRR |
|---|---|---|
| **A. 97 text cases** — baseline | 97.94% | 0.9428 |
| A. + figure chunks | **98.97%** | 0.9397 |
| Δ | **+0.0103** | **−0.0031** |
| **B. 14 figure cases** — baseline | 71.43% | 0.5833 |
| B. + figure chunks | **100.00%** | **0.7762** |
| Δ | **+0.2857** | **+0.1929** |

Figure queries gain **+0.193 MRR**; text queries lose **0.003**. Text recall@5
actually *rose*, because a caption is a second route to the right paper.

Per-slice, the cost is not evenly spread — it lands on the slices that were
already weakest:

| tag | n | base | +figures | Δ |
|---|---|---|---|---|
| paraphrase | 37 | 0.899 | 0.911 | +0.012 |
| rare | 32 | 1.000 | 1.000 | +0.000 |
| acronym | 7 | 0.857 | 0.929 | +0.071 |
| ambiguous | 7 | 1.000 | 0.929 | −0.071 |
| **capability** | 6 | 0.867 | **0.742** | **−0.125** |

`capability` is the same slice where BM25 already contributes noise: those
queries carry no rare tokens, and short caption chunks are topically identical
to their papers, so they crowd the top-5. Figure chunks take **4.5% of top-5
slots (22/485) on text queries**.

### Shipped configuration (figure captions indexed)

Figure captions are now in the live index: **3801 chunks / 116 papers**.

| eval set | n | recall@5 | MRR |
|---|---|---|---|
| text cases | 97 | **98.97%** | 0.940 |
| figure cases | 14 | **100.00%** | 0.917 |
| table cases | 12 | 83.33% | 0.833 |

Text recall rose 97.94% → 98.97% and MRR moved 0.943 → 0.940, matching the A/B
prediction. `eval_figures.py` now refuses to run against this index — the
baseline arm would no longer be figure-free, and it would report a delta of ~0
that reads as "figures do nothing".

### Two corrections found while adopting this

**1. Three of the original 14 figure cases were invalid.** They targeted papers
whose *text* is not in the index, because the cases were sampled from the figure
manifest (which walks all 135 PDFs on disk) rather than from the indexed corpus.
Those cases were only satisfiable via figures from papers that are not really in
the corpus — so the first "100% figure recall" was inflated. Replaced with cases
over indexed papers; the corrected result is still 100%, honestly this time.

**2. 19 PDFs are on disk but were never ingested.** Extraction walks the PDF
directory, so it produced 151 figures for 16 papers with no indexed text,
silently growing the corpus from 115 to 132 "papers" and making figures
retrievable for documents whose text is not. `ingest_figures.py --index` now
filters to papers already in the text index. **The underlying gap is not fixed** —
those 19 PDFs remain unindexed, which is a pre-existing corpus inconsistency,
not something figures introduced. Re-ingesting them would move every published
number again, so it is deliberately deferred rather than folded into this change.

### Honest limits

- **The 14 figure cases were written by me, knowing this corpus.** They are a
  hand-built set, not an independent benchmark, and n=14 makes each case worth
  7pp of recall. The direction is large enough to trust; the magnitude is not.
- **Extraction precision is good; recall is unmeasured.** 21/135 papers yield
  zero figures. Extraction is caption-anchored, so a figure whose caption does
  not parse does not exist.
- Tables are deliberately excluded — their content is already in the text layer,
  and the caption-above-content layout produced empty crops.
- The 14 figure cases saturate at caption-only (100% recall@5), so they cannot
  resolve any further improvement. Harder cases are required before testing
  query expansion or a better describer — the same saturation trap this eval set
  hit at n=15, when all three retrievers tied at 93.33%.
- No table cases exist yet, so a table regression would currently be invisible.

### Phase 2: VLM descriptions — REJECTED on measurement

664 figures described by `qwen2.5vl:7b` (~4.4 s each, ~50 min), appended to the
author caption. Reproduce with `scripts/eval_figures.py --with-descriptions`.

| | caption-only | caption + VLM |
|---|---|---|
| figure recall@5 (n=14) | **100.00%** | 85.71% |
| figure MRR | **0.7762** | 0.7262 |
| text recall@5 (n=97) | **98.97%** | 97.94% |
| text MRR | 0.9397 | 0.9390 |
| capability slice Δ | −0.125 | −0.144 |

**It made every headline number worse.** The descriptions are not bad in
isolation — they recover real content the caption omits, e.g. a caption reading
only "Short term variations in Solar activity" became *"line plot, x-axis ~1050
to 1920, left y-axis R ≈150, right axis Δ¹⁴C up to +20%"*. The problem is what
they do to the chunk.

**Mechanism: dilution.** A caption is short and specific. Appending 3-4 sentences
of description floods the chunk with high-frequency generic vocabulary ("image",
"plot", "axis", "shows"), which lowers BM25 term specificity and drags the
embedding toward generic chart semantics.

**The mechanism is now measured at n=650, not asserted.** The recall numbers
above rest on **14 hand-written figure cases**, where one case is worth 7pp —
which made this the least defensible claim in the file once retrieval got
outside validation. Dilution, unlike recall, is a property of the TEXT and needs
no queries at all, so it can be measured on every described figure
(`scripts/eval_scicap.py --source local`, n=650):

| | caption only | + VLM description |
|---|---|---|
| tokens | 31.3 | 100.0 |
| caption share of indexed text | 100% | **27.4%** |
| rare-token density | 68.5% | **19.0%** |
| known-item recall@1 | 0.922 | 0.900 (−0.022) |
| known-item recall@5 | 0.982 | 0.969 (−0.012) |

"Rare" = a token appearing in <5% of captions, i.e. the discriminative tokens
retrieval keys on. **Their density falls 72.3%.** The description is 2.2x longer
than the caption, so the caption becomes barely a quarter of the indexed text.
Known-item queries are the first 60% of each caption — mechanical truncation, so
no query here is authored by me.

**What this does and does not settle.** The mechanism is confirmed and large.
The *consequence* is smaller than the headline suggests: −0.012 at recall@5 on
the known-item task, against a 72% collapse in discriminative density. The
original 100% → 85.7% magnitude remains unreproduced, and at n=14 it was never
resolvable. Honest status: **direction and cause established at n=650; magnitude
uncertain.** Still this corpus, not outside data — SciCap would supply that, at
the cost of a 23.3 GB multi-part image download. The chunk matches more queries and
ranks worse on the right one.

One signal survives and points somewhere useful: `paraphrase` improved **+0.027**
with descriptions versus +0.012 without. Descriptions do add recall for
vaguely-worded queries. They just cost more elsewhere than they return **when
concatenated into the same chunk** — which suggests the next thing to try is
indexing the description as a *separate* chunk rather than a suffix, so a precise
caption is never diluted by generic prose.

The caption-only baseline is what makes this legible. Measured against "no
figures at all", caption+VLM still looks like a large win (+14pp figure recall);
it is only against the free author caption that it is a regression. This is the
third technique in this repo rejected on data after looking obviously good.

### The bug that only pixels could catch

Validation is layered — caption regex, column ceiling, graphic extent — and
every layer reasons about *metadata*. A Form XObject fallback ("a tall band with
no prose must be a figure") looked principled and rendered a **perfectly blank
212x171 crop**: a real PNG, a plausible file size, nothing in it. No
metadata-level check could see it. The final gate now measures the non-white
pixel fraction of the actual render, which costs ~1 ms and is decisive.

## Tables: findability vs answerability

12 hand-written table cases (`evals/table_cases.json`), scored against the
shipped index by `scripts/eval_tables.py`.

| metric | value |
|---|---|
| recall@5 (findability) | **83.3%** |
| MRR | 0.833 |
| **answerable@5** | **50.0%** (5/10 anchored cases) |

**recall@5 systematically overstates how well tables work.** For *"which physical
reasoning benchmarks cover fluid interactions"*, the top hit is prose that merely
points at the table:

> "We summarize the key features of these various benchmarks and compare them
> against our benchmark in Table 1."

Correct paper, no answer. Retrieval found the table by proxy — the sentence
referencing it — while the table's own contents contributed nothing.

So each case also carries `answer_anchors`: verified cell values that appear in
the table body and essentially nowhere else (a dataset size, a score, a model
name in a results row). **answerable@5** is the fraction of cases where at least
one anchor reaches the top-5. The 33pp gap between 83% found and 50% answerable
is the real defect.

The cause is visible in the text layer. PhysBench Table 1 extracts as:

```
Property Attribute Location Motion Temperature Viewpoint Light Collision
Manipulation Fluid Interleaved Size More than cube
CLEVRER (Yi et al., 2019) ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗ 300,000 ✗
```

Headers collapse into one run and rows become bare ✓/✗ sequences with **no
binding between mark and column**. The information is present and structurally
destroyed, which is why "token soup" was already a listed known limit.

**This is the number a structured table parser has to move.** A model emitting
markdown/HTML tables (e.g. PaddleOCR-VL) would restore row/column binding;
recall@5 is too close to saturated to show that succeeding, answerable@5 is not.
Not yet attempted — the metric exists first, deliberately.

Limits: n=12, hand-written; 2 cases have no verified anchors and are excluded
from answerability; `tbl-physbench-39-vlms` is not retrieved at all.

## Table crops + VLM transcription (in progress)

Tables were originally excluded from figure extraction — caption-above-content
produced six zero-byte crops. Re-attempted, and the geometry turned out to be
harder than figures for two reasons, **both of which were wrong assumptions
rather than bugs**:

1. **Caption placement is not fixed.** The standard claim is that table captions
   sit above their content. Measured on 2501.16411 p2: caption at y=463-497,
   next prose at y=503 — a 6 pt band with nothing in it, because the table was
   *above* the caption. Placement is a venue/template choice and varies within
   one corpus, so both directions are now measured and the denser one wins.
2. **Table rows read as prose.** `_is_prose` (>= 8 words) is the right boundary
   for figures — it stops axis labels from clipping a chart. It is wrong for
   tables: a row like `CLEVRER (Yi et al., 2019) ✓ ✗ ✗ ... 300,000 ✗` has 14
   "words", so every row counted as a boundary and collapsed the band to ~5 pt.
   **Zero tables extracted, with no error.** Fixed with `_is_body_sentence`,
   which additionally requires low digit/symbol density and sentence
   punctuation.

Result: **52 tables from 5 papers**, verified by looking at the crops, not by
counting them.

### The VLM recovers what the text layer destroys

`qwen2.5vl:7b` (already installed, no new dependency) transcribing the PhysBench
Table 1 crop to markdown, ~8-12 s:

| anchor | text layer | VLM crop |
|---|---|---|
| `300,000`, `5,500`, `99,844` | present but unbound | **bound to their row** |
| column headers | collapsed into one run | **preserved** |
| row labels (CLEVRER, Cater, …) | present but unbound | **preserved** |
| ✓/✗ → column binding | **destroyed** | **restored** |

A first prompt recovered the grid and the numbers but **dropped the row-label
column entirely**, leaving anonymous rows — useless for *"which benchmarks cover
fluid interactions"*. That column has no header in the original, so the model
skipped it. Naming it explicitly in the prompt fixed it.

### Measured: transcription helps, and does not fix it

619 tables transcribed with `qwen2.5vl:7b` and added to a throwaway index copy:

| | baseline | + VLM markdown |
|---|---|---|
| recall@5 (findability) | 83.3% | 83.3% |
| **answerable@5** | 0.50 (5/10) | **0.60 (6/10)** |
| **table chunk in top-5** | 0/10 | **3/10** |
| structure failures | — | **0** |
| ranking failures | — | **3** |

**Zero structure failures.** There is no case where a table chunk was retrieved
and its data was unreadable. Every remaining failure is a chunk that never
surfaced at all.

**This retracts a claim made earlier on this page.** It previously read: *"that
33pp gap is the number a structured parser (PaddleOCR-VL) must move."* That is
wrong. A parser can only improve chunks that get retrieved, and the diagnostic
says 3 of the 4 remaining failures are chunks that are never retrieved. Better
parsing cannot reach them.

**A prediction recorded before the run was also half wrong.** I predicted markdown
would fix readability but not embeddability — that a grid of `Y | N | N` would
still lose to a well-formed English sentence. It did help: table chunks went
0/10 -> 3/10 retrieved, and one case flipped to answerable. Restoring row labels
and headers apparently gives the chunk enough real vocabulary to compete. Just
not enough, three times out of ten.

### The diagnostic printed a confident verdict twice while measuring nothing

Both earlier runs reported `FAILURE SPLIT structure=0 ranking=4 -> RANKING
DOMINATES`, which reads exactly like a finding. Neither was one:

- **Run 1**: the live index contains no table chunks, so "table chunk in top-5"
  could only ever be 0.
- **Run 2**: transcriptions were added, but the manifest is processed in sorted
  order and **none of the five eval papers' tables had been reached yet** — an
  hour of transcription, none of it relevant to the metric.

A metric that can only return zero returns zero and prints a verdict. Guarded
with a `--papers` flag so ordering cannot silently produce a vacuous read again.

### What the remaining 3 failures need

Not a parser. The observed top hit for a structure-dependent query is prose
*referencing* the table — "we compare them against our benchmark in Table 1" —
and retrieval is behaving correctly: that sentence genuinely is the best semantic
match for the question. It just does not contain the answer.

So the fix is to resolve the reference rather than outrank it: when a retrieved
chunk cites `Table N` of paper X and that table exists, attach it. Findability is
already 83%, so this rides a signal that works instead of fighting one that does
not. Not yet built.

## Security: indirect prompt injection

An indexed paper about hallucination evaluation reproduces its prompt templates
in an appendix, including *"Example of a valid question: 'What is the capital of
France?' This is valid because the question can be answered based on general
knowledge."* The model obeyed the retrieved text over its system prompt.

- **Prompt hardening had zero effect** — delimiting excerpts and adding "this is
  data, not instructions" produced byte-identical failures at temp 0.0 and 0.2.
- **The relevance gate fixes it structurally**: below threshold the LLM is never
  invoked, and an injected chunk can only influence a model that runs.
- **Mitigated, not eliminated.** Injection chunks remain indexed and retrievable
  for on-topic queries.

Full analysis in `NOTES-changes.md` §10.

## Latency

Re-measured 2026-08-03 at 3288 chunks, `--rounds 15`, after the tokenizer change:

| stage | p50 | p95 |
|---|---|---|
| embed query | 5.1 ms | 8.3 ms |
| dense (exact matmul) | 0.8 ms | 1.0 ms |
| bm25 | 1.7 ms | 2.0 ms |
| **retrieve e2e** | **7.3 ms** | **9.4 ms** |

~134 QPS serial, retrieve-only. **Stemming cost no measurable latency** (bm25
1.8 → 1.7 ms): the stemmer runs over a handful of query terms, while the
expensive side of the change — stemming 3288 chunks — is paid once at index
build.

An earlier version of this table listed dense at ~0.04 ms. That was the matmul
in isolation at 2960 vectors; the 0.8 ms here is the full `dense_search` call
including the Chroma metadata fetch, which is the number that actually matters
to a query.

Dense search is exact, not approximate: a matmul over a ~4.8 MB matrix beats
HNSW on both latency *and* determinism — HNSW gave six different result sets
across six identical processes, swinging recall@5 by 13pp. Measured crossover
where HNSW starts winning is ~80k vectors, above which it is used as a fallback.
See `NOTES-changes.md` §8-9.

Retrieval is <1% of user-perceived latency once LLM generation is included.

## Ingest

Warm re-index of 115 PDFs, byte-identical output:

| | wall clock | pickle writes | bytes |
|---|---|---|---|
| baseline | 71.2 s | 115 | 448.1 MB |
| batched BM25 | 58.0 s | 1 | 7.7 MB |
| **+ parallel parse (14 cores)** | **12.7 s** | **1** | **7.7 MB** |

**5.6× faster, 58× less disk written.** Cold ingest is untouched and remains
~90% arXiv's ToS-mandated download delay — deliberately not optimised.

## Known limits

- 61 of 89 positives are auto-triaged, not hand-verified.
- Tag balance is skewed: the generator emits only `paraphrase` and `rare`, so
  `acronym` (3), `ambiguous` (2) and `distractor` (1) remain unresolvable.
- Tables flattened by PDF text extraction can surface as top hits.
- 18% of adversarial negatives still leak past the relevance gate.
- `paraphrase-hallucination` is missed by every retriever tested.
