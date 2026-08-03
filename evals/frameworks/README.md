# E1 — Framework ablation: hand-rolled vs LlamaIndex vs LangChain

Same corpus, same eval set, same embedding model, same `top_k`/`final_k`/RRF
constant. The only thing varying is who implements hybrid retrieval.

Measured 2026-08-03 · 3288 chunks / 115 papers · 97 positive eval cases ·
`llama-index-core` 0.14.23 · `langchain` 1.3.14 / `langchain-community` 0.4.2

## Results

| implementation | recall@5 | MRR | build | query p50 | query p95 |
|---|---|---|---|---|---|
| hand-rolled (before this ablation) | 96.91% | 0.9359 | 2.8 s warm | 11.2 ms | 20.3 ms |
| LangChain `EnsembleRetriever` | 96.91% | 0.9261 | 6.1 s | 9.9 ms | 18.5 ms |
| LlamaIndex `QueryFusionRetriever` | **98.97%** | 0.9433 | 8.3 s | 33.5 ms | 43.0 ms |
| LlamaIndex, `skip_stemming=True` | 96.91% | 0.9287 | 10.4 s | 33.5 ms | 40.8 ms |
| **hand-rolled + stemming (shipped)** | **97.94%** | **0.9428** | 4.6 s warm | **9.5 ms** | 18.5 ms |

MRR is chunk-level, matching `scripts/eval_recall.py`. See "Two metrics" below.

## The finding

**LlamaIndex's win was its BM25 tokenizer, not its abstraction.**

The first run had LlamaIndex ahead by 2.06pp recall — a big enough gap to be
suspicious of rather than pleased about. The candidate explanations were
contamination (metadata leaking into the indexed text) or a genuine retrieval
difference. Checking `MetadataMode` showed BM25 indexes `EMBED`-mode content,
which the node config had already stripped to body text — no contamination.

The real cause was tokenization:

```
query: "evaluating hallucinations in multimodal models"
  LlamaIndex : ['evalu', 'hallucin', 'multimod', 'model']      # stemmed, stopworded
  arxiv-rag  : ['evaluating', 'hallucinations', 'in', 'multimodal', 'models']
```

`hallucinations` never matched `hallucination`. Rerunning LlamaIndex with
`skip_stemming=True` dropped it to **exactly** 96.91% recall and 0.919
paraphrase recall — the hand-rolled numbers, to the digit. That is the control
that turns a correlation into a cause.

LangChain's BM25 also defaults to `.split()`, and it scored the same 96.91%.
Two frameworks, two different results, one explanation.

**Porting stemming to `arxiv_rag/index.py::_tokenize` captured the gain**:
96.91% → 97.94% recall, 0.9359 → 0.9428 MRR — matching LlamaIndex's MRR (0.9433)
at **3.5× lower query latency**. Dense-only scores were byte-identical before
and after (94.85% / 0.888), confirming only the sparse path moved.

Per-slice, where the gain landed:

| tag | n | before | after |
|---|---|---|---|
| paraphrase (recall) | 37 | 0.919 | **0.946** |
| rare (MRR) | 32 | 0.975 | **1.000** |
| ambiguous (MRR) | 7 | 0.929 | **1.000** |
| acronym (MRR) | 7 | 0.929 | **0.857** ⬇ |
| capability (MRR) | 6 | 0.875 | 0.867 ⬇ |

Two slices regressed. `acronym` is one case of seven, and the mechanism is
plausible — stemming can merge an acronym into an unrelated stem. It is a real
cost of a net-positive change, not noise to hide.

## The honest reading

Do **not** read this as "frameworks are slow and add nothing." Read it as:

- The abstraction cost is real but modest: LangChain was within noise of
  hand-rolled on quality and latency; LlamaIndex cost 3× per-query latency for
  its fusion path.
- **The frameworks' value here was as an oracle, not as a dependency.** Running
  them surfaced a defect in my own tokenizer that six months of eval work had
  not, because my eval only ever compared my code against itself. A good default
  I did not know I was missing was worth more than the framework I would have
  shipped it in.
- Their defaults are landmines in both directions: `QueryFusionRetriever`
  silently defaults to LLM-based query expansion (`num_queries=4`), and
  LlamaIndex nodes embed their own metadata unless excluded. Either would have
  produced a plausible, wrong number.

## Two metrics

`eval_recall.py::_first_relevant_rank` returns the **chunk** index of the first
relevant hit, while its docstring says "paper-level" and the paper-level loop
beneath it is unreachable dead code. So the repo's published MRR is chunk-level;
distill-lab's harness dedups to papers and reports ~0.002 higher on the same
data. `score.py` computes and reports both. Neither is wrong; mixing them across
a comparison table is.

## Friction log

What each framework made harder, recorded while hitting it — the qualitative
half of the ablation.

**LlamaIndex**
1. `QueryFusionRetriever(num_queries=4)` by default calls an LLM to invent query
   variants. That is query expansion, a different technique (separately measured
   and rejected here). Left at the default it would silently benchmark something
   other than hybrid retrieval.
2. Pinning `num_queries=1` is not enough — `__init__` resolves `Settings.llm`
   **eagerly**, so construction raises `ImportError: llama-index-llms-openai
   package not found` even though no generation happens. A pure-retrieval
   benchmark must install an LLM integration or inject `MockLLM()`. The run here
   wraps that mock in a call counter and **aborts if it is ever invoked**.
3. Node metadata is prepended to the embedded text by default;
   `excluded_embed_metadata_keys` must be set or the arXiv id contaminates the
   vector.
4. The RRF constant in `mode="reciprocal_rerank"` is hardcoded, not a
   constructor argument. It happens to be 60, matching `Config.rrf_k` — by luck.

**LangChain**
1. `BM25Retriever` and `FAISS` come from `langchain-community`, which prints
   "being sunset and is no longer actively maintained" on import.
2. `EnsembleRetriever` is not in `langchain.retrievers` — that module was removed
   in 1.x. It now lives in `langchain_classic.retrievers.ensemble`, with no
   deprecation shim pointing there. Textbook hybrid retrieval therefore spans one
   *sunset* package and one named *classic*.
3. The ensemble returns the whole fused union, so `final_k` truncation is the
   caller's job; `k` must be set on each child retriever separately.
4. `c=60` **is** a constructor argument. Point to LangChain over LlamaIndex.

## Reproducing

The framework arms run in an isolated venv, so a framework version bump cannot
break the shipped retriever:

```bash
python3 -m venv .venv-frameworks
.venv-frameworks/bin/pip install llama-index-core llama-index-embeddings-huggingface \
  llama-index-retrievers-bm25 langchain langchain-community langchain-huggingface \
  faiss-cpu rank_bm25 sentence-transformers
```

```bash
.venv/bin/python evals/frameworks/export_corpus.py
.venv/bin/python evals/frameworks/run_handrolled.py
.venv-frameworks/bin/python evals/frameworks/run_llamaindex.py
.venv-frameworks/bin/python evals/frameworks/run_llamaindex.py --no-stem
.venv-frameworks/bin/python evals/frameworks/run_langchain.py
```

`export_corpus.py` is the fairness control: all three arms retrieve over the
identical 3288 chunks produced by this repo's parser, so no result can be an
artifact of a different splitter. `run_handrolled.py` doubles as the validator
for `score.py` — it aborts if the shared scorer fails to reproduce
`eval_recall.py`'s numbers, since a wrong scorer would corrupt every arm
identically and invisibly.
