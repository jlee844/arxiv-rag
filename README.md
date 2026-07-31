# arxiv-rag

Local hybrid RAG over arXiv ML/AI papers. Fetch → section-aware chunk → embed
→ **BM25 + dense (RRF)** → grounded answer via Ollama (Qwen 2.5) or OpenAI.

Built to be measurable: see [EVAL.md](EVAL.md) for recall/MRR and latency at
20 and 106 papers.

**Stack:** arXiv API → PyMuPDF → sentence-transformers (MPS on Apple Silicon) →
ChromaDB + BM25 → Ollama / OpenAI

![Demo: query → retrieved chunks → grounded answer](docs/demo.png)

<sub>Live session: [`docs/demo-session.txt`](docs/demo-session.txt) · vector: [`docs/demo.svg`](docs/demo.svg)</sub>

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
ollama pull qwen2.5:14b      # ~9 GB; fits comfortably on 36 GB unified memory
ollama serve                 # keep running in another terminal
```

Override model anytime: `export OLLAMA_MODEL=qwen2.5:14b`

**OpenAI fallback:** `export OPENAI_API_KEY=...` and `export ARXIV_RAG_BACKEND=openai`

### 3. Ingest

```bash
# Search ingest
.venv/bin/python scripts/ingest.py "vision language model evaluation" --n 30

# Specific IDs (good smoke test)
.venv/bin/python scripts/ingest.py --ids 2605.22903,2306.09265,2406.12384

.venv/bin/python scripts/ingest.py --list
```

### 4. Query

```bash
.venv/bin/python scripts/query.py "POPE object hallucination polling"
.venv/bin/python scripts/query.py "what is the capital of France?"   # should refuse
.venv/bin/python scripts/query.py "POPE object hallucination" --chunks-only
```

### 5. Eval / latency

```bash
.venv/bin/python scripts/eval_recall.py
.venv/bin/python scripts/bench_latency.py --rounds 15
pytest tests/ -v
```

## Architecture

```
arXiv API
   │
   ▼
fetch.py          metadata + PDFs (rate-limited)
   │
   ▼
parse.py          section-aware chunks + Unicode ligature normalize (NFKD)
   │
   ▼
embed.py          all-MiniLM-L6-v2 (MPS when available)
   │
   ├─── index.py  ChromaDB (dense) + BM25Okapi (sparse), both on disk
   │
   ▼
retrieve.py       Reciprocal Rank Fusion + dense_rank / bm25_rank provenance
   │
   ▼
generate.py       excerpt-numbered context; bib [n] stripped from bodies;
                  system prompt refuses when unsupported
```

## Configuration

`arxiv_rag/config.py` or env:

| Setting | Default | Notes |
|---|---|---|
| `embed_model` | `all-MiniLM-L6-v2` | Swap `all-mpnet-base-v2` for quality |
| `chunk_size` / `chunk_overlap` | 512 / 64 words | |
| `top_k` / `final_k` | 8 / 5 | Per-retriever candidates → LLM context |
| `ARXIV_RAG_BACKEND` | `ollama` | or `openai` |
| `OLLAMA_MODEL` | `qwen2.5:14b` | any `ollama list` tag |

Fusion is **RRF ranks only** — there is no dense/BM25 score weight.

## Design decisions

**Why RRF?** BM25 and cosine live on incompatible scales; weighting them is a
calibration trap. RRF fuses ranks. Consensus max ≈ 0.0328; single-retriever
floor ≈ 0.0164.

**Why hybrid?** BM25 wins on rare tokens (`POPE`, `VARCO-VISION`); dense wins
on paraphrase. They disagree often enough that fusion is justified (see EVAL).

**Why strip paper `[17]` cites before generation?** Same syntax as excerpt
`[1]`. Models copy bibliography numbers; stripping bodies leaves only excerpt
headers to cite.

**Why section chunking?** ML papers have strong section structure; windowing
only kicks in inside long sections.

## Known limits

Documented with numbers in [EVAL.md](EVAL.md): one hard paraphrase miss, OOD
queries can still get high RRF (no naive score-gate), PDF tables become token
soup, eval labels lag corpus growth.

## License

[MIT](LICENSE)
