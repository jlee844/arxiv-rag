# arxiv-rag — Build-It-Yourself Plan

A staged walkthrough for writing this system yourself. Each stage is independently
runnable and independently verifiable: you never write more than ~80 lines before
you can prove it works.

**Reference implementation:** the working code is committed at `4c7f66a`. At any
point you can compare your version against it without leaving the terminal:

```bash
git show 4c7f66a:arxiv_rag/parse.py
```

Treat that as the answer key. Look at it *after* you've attempted a stage, not before —
the struggle is where the learning is.

---

## 0. The mental model (read before writing any code)

RAG exists to solve one problem: **an LLM can't answer questions about documents it
never saw during training.** You can't retrain it, and you can't paste 20 PDFs into a
prompt. So you fetch the few paragraphs that matter and paste only those.

Every RAG system is the same five steps:

```
FETCH → CHUNK → EMBED → RETRIEVE → GENERATE
 PDFs    text     vectors   top-k      answer
         pieces             pieces
```

The interesting engineering is almost entirely in **CHUNK** and **RETRIEVE**. Fetch
is plumbing, embed is one library call, generate is a prompt. If you find yourself
spending your time on the other three, you're polishing the wrong surface.

**The one question that drives every design decision in this project:**
*When a user asks a question, what is the smallest piece of text that fully answers it?*

Too small → the chunk lacks context, the model can't tell what it's about.
Too big → you burn context window and dilute the signal with irrelevant sentences.

That tension is the whole game.

### Why hybrid retrieval (the thing that makes this project worth building)

Most tutorial RAG systems use only dense vector search. They fail on a predictable
class of query, and understanding *why* is the main intellectual payoff here.

| Query | Dense (embeddings) | Sparse (BM25) |
|---|---|---|
| "model struggles with multi-property reasoning" | ✅ finds paraphrases | ❌ no keyword overlap |
| "GRPO" | ❌ rare token, weak vector | ✅ exact match |
| "what is LoRA's rank parameter" | ⚠️ partial | ✅ exact match |

Dense embeddings compress meaning into 384 floats. Rare technical tokens — `GRPO`,
`HDBSCAN`, `vLLM` — get smeared into near-neighbours during that compression, because
the model barely saw them in training. BM25 doesn't compress anything; it does exact
token statistics, so rare tokens are exactly where it's *strongest*.

They fail in opposite directions. That's why you run both.

---

## Stage 1 — Fetch (`arxiv_rag/fetch.py`)

**Goal:** given a search string, get paper metadata and PDFs onto disk.

**Write these three:**

```python
@dataclass
class Paper:
    arxiv_id: str; title: str; authors: list[str]; abstract: str
    categories: list[str]; published: str; pdf_url: str; entry_url: str

def fetch_papers(query: str, max_results: int = 50) -> list[Paper]: ...
def download_pdf(paper: Paper, pdf_dir: Path, skip_existing: bool = True) -> Path: ...
def download_all(papers: list[Paper], pdf_dir: Path) -> dict[str, Path]: ...
```

**Design decisions to make yourself:**

1. **Why a `Paper` dataclass instead of passing the raw `arxiv.Result` around?**
   Because it's a boundary. The `arxiv` library's object is theirs and can change
   shape between versions; `Paper` is yours. Every module downstream depends on
   `Paper`, and none of them import `arxiv`. Swapping to Semantic Scholar later
   means rewriting one file.

2. **Rate limiting.** arXiv's ToS asks for ~3s between API calls and a delay between
   PDF downloads. Set `delay_seconds=3.0` on the client and `time.sleep(1.5)` before
   each PDF GET. Skipping this gets you a 403 and it is genuinely rude — they're a
   nonprofit serving free full-text.

3. **`skip_existing=True`.** You will run ingest dozens of times while debugging
   the parser. Re-downloading 20 PDFs each time is slow and abusive. Cache on disk.

4. **`download_all` catches exceptions per-paper.** One malformed PDF should not
   kill a 50-paper ingest. Print and continue.

**Checkpoint — you're done when this prints 5 real titles:**

```bash
.venv/bin/python -c "
from arxiv_rag.fetch import fetch_papers
for p in fetch_papers('vision language model benchmark', max_results=5):
    print(p.arxiv_id, p.title[:60])
"
```

**Pitfall:** `result.entry_id` is a full URL. `arxiv_id` needs `.split('/')[-1]`,
otherwise you get `http:` as your filename and the download writes to a path you
didn't intend.

---

## Stage 2 — Parse & chunk (`arxiv_rag/parse.py`)

**This is the hardest stage and the one that most determines answer quality.** Budget
real time for it. A mediocre retriever over good chunks beats a great retriever over
bad chunks.

**Write:**

```python
@dataclass
class Chunk:
    chunk_id: str; arxiv_id: str; title: str; authors: str
    published: str; section: str; text: str; chunk_index: int

def _split_by_words(text: str, size: int, overlap: int) -> list[str]: ...
def parse_pdf(pdf_path, paper, chunk_size=512, chunk_overlap=64) -> list[Chunk]: ...
```

**Build it in three passes — do not try to write this in one go:**

**Pass A — get text out.** `fitz.open(path)`, loop pages, concatenate `page.get_text()`.
Print it. It will look bad. That's the point — look at the actual mess before designing
around it.

**Pass B — clean the PDF artifacts.** Academic PDFs are typeset for print, not parsing.
Three regexes fix most of it:

```python
re.sub(r"\n{3,}", "\n\n", text)          # collapse blank-line runs
re.sub(r"(\w)-\n(\w)", r"\1\2", text)    # rejoin words hyphenated across line breaks
re.sub(r"\n(?=[a-z])", " ", text)        # a line starting lowercase = wrapped, not new
```

That third one is the clever one. Understand it before you use it: a newline followed
by a lowercase letter is almost always a soft wrap mid-sentence, whereas a newline
followed by a capital may be a real new line. It's a heuristic, and it's wrong
sometimes — that's acceptable here, and knowing *why* it's acceptable is the skill.

**Pass C — chunk with structure.** Naive RAG splits every 512 words and ignores
document structure. Do better, in two tiers:

- **Tier 1: split at section headings.** A chunk that stops at "5. Conclusion" is
  more coherent than one straddling Results and Conclusion. Write `_SECTION_RE` to
  match both `1. Introduction` and bare `Related Work`.
- **Tier 2: window long sections.** A 3000-word Results section is still too big,
  so split it into overlapping 512-word windows.

**Why overlap at all?** Without it, a sentence spanning a chunk boundary is destroyed —
half its meaning in each chunk, retrievable by neither. 64 words of overlap (12.5%)
means any sentence is intact in at least one chunk. The cost is ~12% more storage,
which is nothing.

**Why is the abstract its own chunk?** It's the highest-signal 200 words in the paper —
a human-written summary of the whole thing. Overview queries ("what papers cover X?")
should hit abstracts. Use `paper.abstract` from the arXiv API, *not* the abstract
scraped from the PDF: the API version is clean text, the PDF version has column
artifacts and footnote markers.

**Why `chunk_size` in words, not tokens?** Words are free to count; tokens need the
tokenizer loaded. ~0.75 words/token, so 512 words ≈ 384 tokens — comfortably under
MiniLM's 512-token limit. This is a deliberate approximation and you should be able
to say why it's safe.

**Checkpoint:**

```bash
.venv/bin/python -c "
from arxiv_rag.fetch import fetch_papers, download_pdf
from arxiv_rag.parse import parse_pdf
from pathlib import Path
p = fetch_papers('vision language model benchmark', 1)[0]
path = download_pdf(p, Path('data/pdfs'))
chunks = parse_pdf(path, p)
print(f'{len(chunks)} chunks')
print(sorted({c.section for c in chunks}))
print(chunks[0].text[:300])
"
```

**Read the output like a reviewer.** Are sections detected, or is everything `body`?
Is chunk 0 the abstract? Is the text readable prose or hyphenation soup? Iterate on
the regexes until you'd be happy retrieving these.

**Pitfall:** `re.split` with capture groups returns the delimiters interleaved in the
result list. That's *why* the section loop re-tests each part with `_SECTION_RE.match`
to decide "is this a heading or content?" If you don't understand this, your sections
will silently be wrong.

---

## Stage 3 — Embed (`arxiv_rag/embed.py`)

**The easiest stage.** ~30 lines. Two functions:

```python
@lru_cache(maxsize=1)
def _model(model_name: str): ...
def embed_texts(texts, model_name=..., batch_size=64, show_progress=False) -> np.ndarray: ...
def embed_query(query, model_name=...) -> np.ndarray: ...
```

**Three decisions worth understanding:**

1. **`@lru_cache` on the loader.** `SentenceTransformer(...)` reads ~90 MB from disk
   and takes 2–3 seconds. You call `embed_texts` once per batch — dozens of times per
   ingest. Without the cache you'd reload the model every call and ingest would take
   minutes instead of seconds. This one decorator is the difference.

2. **`normalize_embeddings=True`.** After L2 normalization, cosine similarity *is*
   the dot product — the expensive division drops out. Chroma's `hnsw:space=cosine`
   then works correctly and faster. Nearly free, so always on.

3. **Import `sentence_transformers` inside the function, not at module top.** It
   pulls in torch (~2s). Modules that only need `Config` or `Chunk` shouldn't pay
   that. Lazy import keeps your CLI's `--help` instant.

**Why `all-MiniLM-L6-v2`?** 22 MB, 384 dims, ~1000 sentences/sec on CPU, no GPU. The
upgrade path is `all-mpnet-base-v2` (768 dims, ~5% better recall, ~2x slower). Start
small — you can re-embed later, and you should not optimize retrieval quality before
you can measure it.

**Checkpoint — verify the geometry, not just that it runs:**

```bash
.venv/bin/python -c "
from arxiv_rag.embed import embed_texts
import numpy as np
v = embed_texts(['transformer attention', 'self-attention mechanism', 'banana bread'])
print('shape', v.shape, 'norm', np.linalg.norm(v[0]))
print('related  ', v[0] @ v[1])
print('unrelated', v[0] @ v[2])
"
```

Norm must be ~1.0. Related pair must score well above the unrelated pair. If it
doesn't, nothing downstream can work and you'd never know from a stack trace.

---

## Stage 4 — Index (`arxiv_rag/index.py`)

**Goal:** one `PaperIndex` class owning two indexes that must never drift apart.

```python
class PaperIndex:
    def add_chunks(self, chunks, batch_size=64) -> int: ...
    def dense_search(self, query_embedding, k=10) -> list[dict]: ...
    def bm25_search(self, query, k=10) -> list[dict]: ...
    def count(self) -> int: ...
```

**The central design problem:** ChromaDB persists itself automatically. BM25 does not —
`rank_bm25` is a pure in-memory object. If you don't handle this, your index works
in the session that built it and silently returns zero BM25 results forever after.
So you pickle the BM25 object and JSON the corpus alongside it, and reload both in
`__init__`.

**Why both files?** The pickle is the scorer; the JSON is the human-readable corpus
you can inspect when retrieval misbehaves. Pickle alone would be opaque, and you
*will* need to debug this.

**Make `add_chunks` idempotent.** Query Chroma for the incoming `chunk_id`s, filter
out ones already present, embed only the remainder. You will re-run ingest constantly;
without this you get duplicate chunks, which corrupt BM25's document-frequency
statistics and quietly degrade ranking.

**Return a uniform dict from both search methods** — `{chunk_id, text, score, **metadata}`.
The fusion layer in Stage 5 must not know or care which retriever produced a result.
Get this contract right and Stage 5 is 40 easy lines.

**Checkpoint:**

```bash
.venv/bin/python -c "
from arxiv_rag.index import PaperIndex
ix = PaperIndex()
print('chunks:', ix.count())
print('bm25:', [r['title'][:40] for r in ix.bm25_search('benchmark', k=3)])
"
```

Then **run it twice in separate processes.** Both must return results. That's the
persistence bug, and it's the one people ship.

---

## Stage 5 — Hybrid retrieval (`arxiv_rag/retrieve.py`)

**The intellectual core. ~40 lines, and the most important 40 in the project.**

```python
_RRF_K = 60
def retrieve(query: str, index: PaperIndex, config=None) -> list[dict]: ...
```

**The problem:** BM25 returns unbounded positive scores (0 to ~30, corpus-dependent).
Cosine similarity returns roughly 0 to 1. **You cannot add these.** Try it and BM25
dominates purely because its numbers are bigger — a scale artifact, not a relevance
signal.

**Two ways out:**

- *Normalize then weight:* min-max each score list to [0,1], then `0.65*dense + 0.35*sparse`.
  Requires tuning that weight per corpus, and min-max is unstable when one list is
  nearly uniform.
- *Reciprocal Rank Fusion:* **throw the scores away and use only the ranks.**

```
RRF(d) = Σ  1 / (k + rank_i(d))     over each retriever i, k = 60
```

RRF wins because scale-invariance is free — ranks are ranks, so no calibration and no
tuning. A doc ranked #1 by both retrievers scores `1/61 + 1/61 = 0.0328`. A doc ranked
#1 by one and absent from the other scores `0.0164`. **Agreement across retrievers is
rewarded automatically**, which is exactly the behaviour you want and never had to
hand-tune.

**What is `k=60` doing?** It flattens the curve. Without it, rank 1 (`1/1`) would be
worth twice rank 2 (`1/2`) — far too peaked, letting one retriever's top hit dominate.
With k=60, ranks 1 and 2 differ by under 2%, so *consensus* matters more than any
single retriever's confidence. 60 is the value from the original RRF paper and is
a fine default. Try 1 and 1000 and watch the rankings change — that's the exercise.

**Checkpoint — the query that proves hybrid beats dense-only.** Find a rare technical
token in your corpus and query it. Compare `index.dense_search` alone against
`retrieve()`. If hybrid doesn't win on rare tokens, your BM25 path is broken —
most likely tokenization (`query.lower().split()` must match how you tokenized the
corpus, or scores are meaningless).

---

## Stage 6 — Generate (`arxiv_rag/generate.py`)

**Goal:** retrieved chunks in, grounded answer out.

```python
_SYSTEM_PROMPT = "..."
def _build_context(chunks: list[dict]) -> str: ...
def generate(query, chunks, config=None, stream=True) -> str: ...
```

**The system prompt is the safety mechanism, and it needs three things:**

1. *"Answer using ONLY the excerpts below"* — scopes the model to your context.
2. *"Cite the paper title in brackets"* — makes claims auditable. Without citations
   you cannot tell grounded answers from hallucinations.
3. *"If the excerpts don't contain enough information, say so"* — **the most important
   line.** Without an explicit escape hatch, models pattern-match toward answering
   anyway. With it, they'll decline.

You can watch this work: the reference implementation, asked about cross-property
reasoning, replied that it couldn't find a benchmark for it rather than inventing one.
That's line 3 doing its job, and it's the difference between a demo and a tool.

**Number your context blocks `[1]`, `[2]`** so citations are checkable against the
retrieval output printed above the answer.

**Backends:** Ollama default (local, free, private), OpenAI behind an env var. Keep
the branch at the top of `generate()` and give both the same signature — one prompt,
two transports.

**Checkpoint:** ask something your corpus definitely can't answer ("what is the
capital of France?"). A correct system *refuses*. If it answers, your prompt is too weak.

---

## Stage 7 — CLIs (`scripts/ingest.py`, `scripts/query.py`)

Thin wiring only. All logic lives in the package; these just parse args and print.

- `ingest.py <query> --n 20` → fetch → download → parse → `add_chunks` → report count
- `query.py [question]` → REPL if no arg, one-shot if given

**Print the retrieved chunks before the answer.** Non-negotiable for a RAG system:
when an answer is wrong you must be able to see instantly whether retrieval failed
or generation failed. Those have completely different fixes, and without visible
chunks you're guessing.

---

## Stage 8 — Tests

Test the pure functions — the ones with no network, no model, no disk:

- `_split_by_words`: correct count, correct overlap, short text doesn't split
- RRF: monotonic (higher rank → higher score)
- `Config`: invariants like `chunk_overlap < chunk_size`, `final_k <= top_k`

Skip integration tests for now. The fast, deterministic tests are the ones you'll
actually run while iterating.

---

## Four real bugs in the reference implementation — fix them as exercises

These are in the committed code. I found them reviewing the run. Fixing them yourself
is better practice than any toy exercise, because they're the kind of thing that
actually ships.

1. **Dead config knob.** `Config.dense_weight = 0.65` is never read — `retrieve.py`
   uses RRF, which has no weight term. A test even asserts on it (`test_config_defaults`),
   which is worse: a test guarding a value nothing uses. *Decide:* delete it, or
   implement weighted fusion as an alternative mode behind a flag?

2. **Empty-index crash.** `index.py:96` passes `n_results=min(k, self._col.count())`.
   On a fresh index that's `n_results=0`, which Chroma rejects. *Fix:* early-return `[]`
   when `count() == 0`.

3. **References sections pollute retrieval.** A bibliography chunk was retrieved at
   rank 2 in the live run — it matched on author surnames, not content. *Fix:* skip
   chunks whose section is `References` at index time. *Then ask:* should this be a
   hard filter or a score penalty? What if someone legitimately asks "who cites X?"

4. **Telemetry noise.** Four `Failed to send telemetry event` lines print on every
   run despite `anonymized_telemetry=False` — a chromadb 0.5.20 / posthog signature
   mismatch. Cosmetic, but it's on every single invocation. *Fix:* pin a different
   chromadb, or suppress the logger.

---

## Suggested order

| Session | Stages | Outcome |
|---|---|---|
| 1 | 0, 1 | PDFs on disk |
| 2 | 2 | Clean chunks with sections — **the long one** |
| 3 | 3, 4 | Both indexes persist across processes |
| 4 | 5 | Hybrid retrieval beating dense-only on rare tokens |
| 5 | 6, 7 | End-to-end grounded answers |
| 6 | 8 + bugs | Tests green, four bugs fixed |

**The rule that makes this work:** never move to the next stage until the current
checkpoint passes. Every stage has a runnable one-liner for exactly this reason.
RAG failures are silent — bad chunks don't crash, they just quietly return
mediocre answers three stages later, and by then you can't tell which stage caused it.

---

## Extensions once it works

- **Reranking:** run a cross-encoder (`ms-marco-MiniLM-L-6-v2`) over the top 20 and
  keep 5. Usually the single biggest quality win available.
- **Query expansion:** have the LLM rewrite the query before retrieval.
- **Evaluation:** 20 hand-written question/expected-paper pairs, measure recall@5.
  Do this *before* tuning anything — otherwise you're optimizing on vibes.
- **Incremental ingest:** `--since 2026-01-01` to pull only new papers.
