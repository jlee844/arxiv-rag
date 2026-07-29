# arxiv-rag

Local RAG (Retrieval-Augmented Generation) over arXiv ML/AI papers. Runs fully on CPU — no GPU required.

**Stack:** arXiv API → PyMuPDF → sentence-transformers → ChromaDB + BM25 → Ollama (local LLM)

```
Query: "what methods reduce hallucination in VLMs?"

[1] PercepTax (2025-06) — Cross-Property Reasoning
    Experiments show that GPT-5 achieves only 39.84% on cross-property
    reasoning compared to 87% on object description...

[2] VLM Hallucination Survey (2024-11) — Methods
    Contrastive decoding, RLHF-based calibration, and retrieval-augmented
    generation have emerged as the dominant approaches...

Answer [llama3.2:3b]:
Based on the papers, three main approaches address VLM hallucination: [PercepTax]
identifies that the bottleneck is cross-property reasoning integration, not
individual property recognition. [VLM Hallucination Survey] covers contrastive
decoding and RLHF calibration as the most effective interventions...
```

## Quickstart

### 1. Install dependencies

```bash
cd arxiv-rag
pip install -r requirements.txt
```

### 2. Install Ollama (local LLM, no GPU needed on Apple Silicon)

```bash
# macOS
brew install ollama

# Then pull a small model (~2GB)
ollama pull llama3.2:3b

# Start the server (keep this running in a separate terminal)
ollama serve
```

> **No GPU?** `llama3.2:3b` runs at ~8 tok/s on Intel Mac CPU, ~30 tok/s on Apple Silicon via Metal.  
> **OpenAI fallback:** `export OPENAI_API_KEY=sk-... && export ARXIV_RAG_BACKEND=openai`

### 3. Ingest papers

```bash
# Ingest 30 papers on VLM evaluation
python scripts/ingest.py "vision language model evaluation benchmark" --n 30

# Ingest papers on RL for LLM agents
python scripts/ingest.py "reinforcement learning LLM agent uncertainty" --n 20

# Add specific papers by arXiv ID
python scripts/ingest.py --ids 2406.01234,2501.16411

# Check what's indexed
python scripts/ingest.py --list
```

### 4. Query

```bash
# Interactive mode
python scripts/query.py

# One-shot
python scripts/query.py "how does GRPO compare to PPO for LLM fine-tuning?"

# See retrieved chunks without generating
python scripts/query.py "LoRA fine-tuning surgical robotics" --chunks-only
```

## Architecture

```
arXiv API
   │
   ▼
fetch.py          Download paper metadata + PDFs (respects rate limits)
   │
   ▼
parse.py          Section-aware chunking
                  • Abstract → dedicated chunk
                  • Body → split at section headers, then 512-word windows
                    with 64-word overlap
   │
   ▼
embed.py          all-MiniLM-L6-v2 (22 MB, CPU-fast, L2-normalized)
   │
   ├─── index.py  ChromaDB (dense, cosine) + BM25Okapi (sparse, keyword)
   │              Both persist to data/ — survives restarts
   │
   ▼
retrieve.py       Hybrid: RRF fusion of dense + BM25 results
                  • BM25 wins on exact terms: "GRPO", "LoRA", "HDBSCAN"
                  • Dense wins on semantic similarity
                  • RRF is robust to score scale differences
   │
   ▼
generate.py       System prompt grounds LLM to retrieved context only
                  Ollama (local) or OpenAI (OPENAI_API_KEY)
```

## Configuration

All tuneable in `arxiv_rag/config.py` or via env vars:

| Setting | Default | Notes |
|---|---|---|
| `embed_model` | `all-MiniLM-L6-v2` | Swap to `all-mpnet-base-v2` for ~5% better recall |
| `chunk_size` | 512 words | ~384 tokens |
| `chunk_overlap` | 64 words | |
| `top_k` | 8 | Candidates per retriever before fusion |
| `final_k` | 5 | Chunks passed to LLM |
| `ARXIV_RAG_BACKEND` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3.2:3b` | Any model in `ollama list` |

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Design decisions

**Why RRF over weighted score normalization?**  
BM25 scores and cosine similarity live on different scales. Normalizing them requires per-query calibration. RRF rank fusion is scale-invariant and consistently performs well without tuning.

**Why section-level chunking first?**  
ML papers have strong section structure (Abstract, Related Work, Method, Experiments). Splitting at section boundaries keeps semantically coherent text together. Word-count windowing only activates for long sections.

**Why `all-MiniLM-L6-v2` not a larger model?**  
On CPU without quantization, MiniLM-L6 encodes ~1000 chunks/min. `all-mpnet-base-v2` (5× slower) gives ~5% better retrieval recall — not worth the indexing time for interactive use.

**Why Ollama over llama.cpp directly?**  
Ollama handles model download, Metal acceleration on Apple Silicon, and a persistent server with an OpenAI-compatible API — less setup for the same result.
