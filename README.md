# arxiv-rag

Local hybrid RAG over arXiv ML/AI papers. Fetch → section-aware chunk → embed →
**BM25 + dense (RRF)** → relevance gate → grounded answer via Ollama (Qwen 2.5)
or OpenAI. Ships with a streaming HTTP API and a zero-build web UI.

Built to be measurable. Every claim below is reproducible with a command in this
repo; the numbers live in [EVAL.md](EVAL.md).

![Demo: query → retrieved chunks → grounded answer](docs/demo.png)

<sub>Live session: [`docs/demo-session.txt`](docs/demo-session.txt) · vector: [`docs/demo.svg`](docs/demo.svg)</sub>

## Does hybrid retrieval actually help?

Ablated over 97 scored queries, 115 papers / 3288 chunks:

| retriever | recall@5 | MRR | paraphrase (37) | rare token (32) | capability (6) |
|---|---|---|---|---|---|
| dense only | 94.85% | 0.888 | 89% | 97% | **0.889** |
| BM25 only | 91.75% | 0.862 | 81% | 100% | 0.611 |
| **hybrid RRF** | **97.94%** | **0.943** | **95%** | **100%** | 0.867 |

```bash
.venv/bin/python scripts/eval_recall.py --ablate
```

Dense wins paraphrase, BM25 wins rare technical tokens, fusion takes both. That
split is the entire argument for hybrid, and it only became visible after the
eval set grew from 15 to 76 cases — at n=15 all three tied at 93.33%, because
one case was worth 6.7pp and the effect being measured is 2.6pp.

The `capability` slice is where that argument stops holding: those queries name
an ability ("physical properties like mass and friction") while every candidate
paper is topically identical, so BM25 has nothing to match (MRR 0.611) and RRF
carries that noise into hybrid — the one slice where fusion scores **below**
dense alone. See [EVAL.md](EVAL.md).

**Where those numbers came from is itself a result.** Hybrid sat at 96.91% /
0.936 until benchmarking this repo against LlamaIndex and LangChain showed
LlamaIndex ahead — and the entire gap turned out to be that its BM25 tokenizer
*stems* while mine called `text.lower().split()`, so `hallucinations` never
matched `hallucination`. Porting stemming captured the gain at 3.5× lower
latency than the framework. Details in
[evals/frameworks/](evals/frameworks/README.md).

**Stack:** arXiv API → PyMuPDF → sentence-transformers (MPS on Apple Silicon) →
ChromaDB + BM25 → FastAPI (SSE) → Ollama / OpenAI

## Quickstart

### 1. Environment

```bash
cd arxiv-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama (default backend)

```bash
brew install ollama          # macOS
ollama pull qwen2.5:14b      # ~9 GB
ollama serve                 # keep running in another terminal
```

**OpenAI fallback:** `export OPENAI_API_KEY=...` and `export ARXIV_RAG_BACKEND=openai`

### 3. Ingest

```bash
.venv/bin/python scripts/ingest.py "vision language model evaluation" --n 30
.venv/bin/python scripts/ingest.py --ids 2605.22903,2306.09265,2406.12384
.venv/bin/python scripts/ingest.py --list
```

Parses in parallel across all cores and rebuilds BM25 once per run — 12.7 s to
re-index (measured at 109 PDFs; not re-run since the corpus grew to 115).

### 4. Query — CLI

```bash
.venv/bin/python scripts/query.py "POPE object hallucination polling"
.venv/bin/python scripts/query.py "what is the capital of France?"   # refuses
```

### 5. Serve — API + web UI

```bash
.venv/bin/uvicorn arxiv_rag.api:app --reload --port 8001
```

Open <http://localhost:8001>. Interactive API docs at `/docs`.

| endpoint | purpose |
|---|---|
| `GET /api/health` | index size, models, whether exact search is active |
| `POST /api/search` | retrieval only — fast, and the best debugging surface |
| `POST /api/chat` | SSE stream: `sources` → `token`… → `done` |

```bash
curl -s localhost:8001/api/health | python3 -m json.tool
curl -s -X POST localhost:8001/api/search -H 'Content-Type: application/json' \
  -d '{"query":"CLIP","mode":"hybrid","k":3}'
curl -N -X POST localhost:8001/api/chat -H 'Content-Type: application/json' \
  -d '{"query":"how is POPE used to evaluate hallucination?"}'
```

`mode` accepts `hybrid` | `dense` | `bm25`, so the ablation is switchable live
in the browser. Every result carries `dense_rank` / `bm25_rank` / `rrf_score`,
which makes fusion visible rather than asserted.

### 6. MCP server

Exposes retrieval as tools any MCP client can call — no Chroma, BM25, or
embedding model needed on the caller's side.

```bash
.venv/bin/python -m arxiv_rag.mcp_server        # stdio
```

Register it with a client — see [`mcp.json.example`](mcp.json.example), or:

```bash
claude mcp add arxiv-rag -- "$(pwd)/.venv/bin/python" -m arxiv_rag.mcp_server
```

| tool | returns |
|---|---|
| `search_papers(query, k, mode)` | excerpts + `dense_rank`/`bm25_rank`/`rrf_score` provenance |
| `list_papers()` | every indexed arXiv id + title |
| `search_figures(query, k)` | matching charts/plots + a path to the rendered PNG |
| `index_status()` | chunk/paper counts, models, whether exact search is active |

**The trust boundary is the interesting part.** This repo found a live indirect
prompt injection inside an indexed paper, and fixed it *structurally* — a
relevance gate that refuses before the LLM is invoked. **An MCP server cannot
reuse that defence**, because it has no LLM of its own: it hands corpus text to
the caller's model.

So the server does the only three honest things available to it:

1. **Labels the boundary** — every excerpt is wrapped in explicit
   `UNTRUSTED PAPER EXCERPT` markers. A hint, not a guarantee.
2. **Reports the gate instead of enforcing it** — `search_papers` returns
   `relevance` and `below_relevance_gate`, so the caller can apply the same
   policy this repo does. Verified: the injection string *"what is the capital
   of France?"* scores 0.2277 against a 0.35 gate and is flagged.
3. **Cannot act** — read-only over a local index. No tool here can follow an
   instruction found in a paper.

It does not *fix* injection. It makes the boundary visible and hands the caller
the same signal. That is the ceiling for a retrieval server.

### 7. Framework ablation (hand-rolled vs LlamaIndex vs LangChain)

The same hybrid pipeline, built three ways, scored on the same 97 cases over
byte-identical chunks:

| implementation | recall@5 | MRR | query p50 |
|---|---|---|---|
| LangChain `EnsembleRetriever` | 96.91% | 0.9261 | 9.9 ms |
| LlamaIndex `QueryFusionRetriever` | 98.97% | 0.9433 | 33.5 ms |
| LlamaIndex, stemming disabled | 96.91% | 0.9287 | 33.5 ms |
| **hand-rolled (this repo)** | **97.94%** | **0.9428** | **9.5 ms** |

That third row is the point: turning off LlamaIndex's BM25 stemming drops it to
*exactly* the hand-rolled score, so its lead was a tokenizer default, not an
abstraction. The frameworks earned their keep here **as an oracle, not as a
dependency** — they exposed a defect my own eval could never surface, because it
only ever compared my code against itself.

Full method, per-slice results, and the friction log (LlamaIndex's fusion
retriever requires an LLM even with generation disabled; LangChain's hybrid
retriever now spans a *sunset* package and one named *classic*) are in
[evals/frameworks/README.md](evals/frameworks/README.md).

### 8. Multimodal — figure retrieval

Extracts charts, plots, and diagrams from the PDFs and makes them retrievable.

```bash
.venv/bin/python scripts/ingest_figures.py --index   # extract + index captions
.venv/bin/python scripts/eval_tables.py              # findability vs answerability
.venv/bin/python scripts/describe_figures.py         # VLM captions (measured worse)
```

Shipped: **3801 chunks / 116 papers** — text recall@5 **98.97%**, figure cases
**100%**, table cases 83.3%.

**Why clip-render instead of extracting images.** `page.get_images()` finds 462
embedded bitmaps in a 25-paper sample and **misses 73 pages of vector figures** —
matplotlib and TikZ plots are PDF drawing operators, not bitmaps, so the ablation
charts you most want return nothing and no error. Rendering the figure *region*
captures vector and raster identically.

**The measured trade**, scored on both the 97 text cases and 14 figure cases:

| | recall@5 | MRR |
|---|---|---|
| 97 text cases, baseline | 97.94% | 0.9428 |
| 97 text cases, + figures | 98.97% | 0.9397 |
| 14 figure cases, baseline | 71.43% | 0.5833 |
| **14 figure cases, + figures** | **100.00%** | **0.7762** |

Figure queries gain **+0.193 MRR**; text queries lose **0.003** and *gain* 1pp of
recall. The cost is not evenly spread — it lands on `capability` (−0.125), the
slice where BM25 already contributes noise.

**VLM descriptions were measured and rejected.** Captioning all 664 figures with
`qwen2.5vl:7b` and appending the result to the author caption made every headline
number *worse* — figure recall@5 100% → 85.7%, figure MRR 0.776 → 0.726. The
descriptions are accurate; the problem is dilution, since generic prose
("image", "plot", "axis") lowers the caption's term specificity. Kept behind
`--with-descriptions` so the negative result stays reproducible. Details and
limits in [EVAL.md](EVAL.md); the 14 figure cases are hand-written and small, so
trust the direction more than the magnitude.

### 9. Eval / latency / tests

```bash
.venv/bin/python scripts/eval_recall.py --ablate    # retriever comparison
.venv/bin/python scripts/eval_recall.py --gate      # abstain threshold sweep
.venv/bin/python scripts/bench_latency.py --rounds 15
pytest tests/ -v
```

## Architecture

```
arXiv API
   │
   ▼
fetch.py          metadata + PDFs (ToS rate-limited, cached on disk)
   │
   ▼
parse.py          font-metric section detection, NFKD ligature fix,
                  parallel across cores
   │
   ▼
embed.py          all-MiniLM-L6-v2 (MPS when available)
   │
   ├─── index.py  ChromaDB + exact matmul (dense) · BM25Okapi (sparse)
   │              batched rebuild; both persisted to disk
   ▼
retrieve.py       Reciprocal Rank Fusion + dense_rank / bm25_rank provenance
   │
   ▼   relevance gate — below Config.min_relevance the LLM is never invoked
   │
generate.py       excerpt-numbered context; refuses when unsupported
   │
   ▼
api.py            FastAPI + SSE streaming, citation check, static web UI
```

## Configuration

`arxiv_rag/config.py` or env:

| Setting | Default | Notes |
|---|---|---|
| `embed_model` | `all-MiniLM-L6-v2` | swap `all-mpnet-base-v2` for quality |
| `embed_device` | auto | pin `cpu` for reproducible measurement |
| `chunk_size` / `chunk_overlap` | 512 / 64 words | |
| `top_k` / `final_k` | 8 / 5 | per-retriever candidates → LLM context |
| `rrf_k` | 60 | RRF damping; higher rewards consensus |
| `min_relevance` | 0.35 | abstain gate on dense cosine |
| `exact_search_max` | 80 000 | above this, fall back to HNSW |
| `rerank` | `False` | cross-encoder; **measured worse**, see below |
| `ARXIV_RAG_BACKEND` | `ollama` | or `openai` |
| `OLLAMA_MODEL` | `qwen2.5:14b` | any `ollama list` tag |

## Design decisions

**Why RRF, not weighted scores?** BM25 is unbounded and corpus-relative; cosine
is 0–1. Weighting them is a calibration trap that has to be retuned whenever the
corpus changes. RRF fuses ranks, so it needs no calibration.

**Why exact search instead of the vector DB's ANN index?** ChromaDB's HNSW
returned **six different result sets across six identical processes**, silently
swinging eval recall@5 by 13pp. At this corpus size a matmul over a 4.4 MB
matrix is both deterministic and *faster* (0.04 ms vs 1.1 ms). HNSW is kept as a
fallback above the measured ~80k-vector crossover.

**Why a relevance gate rather than a better prompt?** An indexed paper about
hallucination evaluation contains prompt templates in its appendix — including
the literal line *"Example of a valid question: 'What is the capital of France?'
… can be answered based on general knowledge."* The model obeyed the retrieved
text over its system prompt: indirect prompt injection, arriving organically.
Hardening the prompt changed the output **byte-for-byte not at all**. Gating on
retrieval similarity works because an injected chunk can only influence a model
that gets invoked.

**Why is reranking off?** It was implemented, measured, and rejected: MRR
0.900 → 0.867, abstain AUC 0.970 → 0.927, +82 ms/query, and the known failure
case unchanged. Kept behind a flag so the negative result stays reproducible
(`--rerank`).

**Why font-based section detection?** A regex matched table rows and figure
captions as headings, which split real sections and silently discarded content.
Typography separates structure from subject matter; character patterns don't.

**Why strip paper `[17]` cites before generation?** Same syntax as excerpt
`[1]`. Models copy bibliography numbers; stripping them from bodies leaves only
excerpt headers to cite.

## Known limits

Documented with numbers in [EVAL.md](EVAL.md):

- 61 of 97 positive eval cases are auto-triaged, not hand-verified.
- **32% of adversarial negatives (7/22) leak past the relevance gate** at the
  shipped 0.35 threshold. Positives and negatives overlap — max negative 0.623,
  min positive 0.357 — so no threshold separates them cleanly. (An earlier
  README said 18%; that is the leak rate at 0.40, not at the shipped default.
  Raising to 0.40 buys it at the cost of 1 false abstain: `--gate` prints the
  sweep.)
- One hard paraphrase case is missed by every retriever tested.
- Stemming the BM25 tokenizer cost `acronym` MRR (0.929 → 0.857, one case of
  seven) to buy `paraphrase` and `rare`. Net positive, not free.
- PDF tables flatten into token soup and can surface as top hits.

## License

[MIT](LICENSE)
