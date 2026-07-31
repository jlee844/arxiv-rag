# Evaluation

Measured on Apple Silicon (M4 Max, 36 GB). Retrieval only unless noted.
Cases: `evals/retrieval_cases.json` · runner: `scripts/eval_recall.py` ·
latency: `scripts/bench_latency.py`.

```bash
.venv/bin/python scripts/eval_recall.py
.venv/bin/python scripts/bench_latency.py --rounds 15
```

## Retrieval quality

Easy title-ish queries hit **100% recall@5** on a small corpus — that only
proves plumbing. The hard tagged set is the signal:

| corpus | recall@5 | MRR | paraphrase MRR |
|---|---|---|---|
| 20 papers / 755 chunks | 93.33% (14/15) | 0.856 | 0.722 |
| 106 papers / 2834 chunks | 93.33% (14/15) | **0.900** | **0.833** |

Acronym / ambiguous / rare / distractor tags: 100% recall at both scales.
Hybrid RRF held under ~5× paper growth; paraphrase improved.

**Known miss:** `paraphrase-hallucination` — *"datasets that check whether VLMs
invent objects that are not in the image"* never retrieves `2605.22903`. Dense
paraphrase fails on that wording; BM25 has no rare token to latch onto.

### Out-of-distribution queries

RRF always returns top-k. OOD top scores are **not** a reliable abstain signal:

| query | N=20 top RRF | N=106 top RRF |
|---|---|---|
| capital of France | 0.03078 (≈ consensus max) | 0.01639 (floor) |
| stock price | 0.01639 | 0.03202 (near max) |
| cooking | 0.01639 | 0.03279 (theoretical max) |

A flat RRF threshold is unsafe. Grounding relies on the generation system
prompt (refusal when excerpts don't support an answer).

## Latency (retrieve only)

| stage | 755 chunks p50 | 2834 chunks p50 |
|---|---|---|
| embed | 4.8 ms | 5.1 ms |
| dense | 1.0 ms | 1.1 ms |
| bm25 | 0.7 ms | 1.8 ms |
| retrieve e2e | 6.3 ms | 8.2 ms |
| approx QPS | ~158 | ~121 |

Embed dominates. BM25 is the only stage that scales with N so far. Interactive
UX cost is local LLM generation (Ollama), not retrieval.

## Known limits (not blocking)

- Tables flattened by PDF text extract can surface as top hits.
- Models may invent sibling names for stripped paper bibliography slots.
- Eval labels still center on the original ~20 papers — optimistic as the
  corpus grows; expand cases before claiming broader recall.
- No cross-encoder rerank / query expansion yet (see PLAN extensions).
