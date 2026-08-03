# arxiv-rag v3 — serve it, ship it, deploy it

Successor to `PLAN-v2.md`. v2's Phase A (ablation) is done and produced a
larger finding than the ablation itself; see `NOTES-changes.md` §8–9.

**Port note:** this project uses **8001**. Port 8000 is held by the
`movement_care` FastAPI backend.

---

## Where things stand

| | |
|---|---|
| corpus | 115 papers / 3288 chunks |
| retrieval | hybrid RRF; dense = exact matmul, deterministic |
| ablation | dense 94.85% · BM25 91.75% · **hybrid 97.94%** recall@5 (MRR .888/.862/**.943**) — post-stemming, see evals/frameworks/ |
| eval set | **119 cases** — 97 positive (61 auto-triaged, 21 hand-added) + 22 negative |
| safety | cosine relevance gate + citation check; injection mitigated, 32% negatives leak at the shipped 0.35 gate (18% at 0.40) |
| latency | retrieve 7.3 ms p50 · generation seconds (qwen2.5:14b) |
| API | **shipped** — SSE streaming, `/api/{health,search,chat}`, web UI on :8001 |
| ingest | cold ~190 s (~90% ToS sleep) · **warm re-index 12.7 s** (was 71.2 s) |
| rejected | cross-encoder rerank · prompt hardening · mpnet · larger top_k |

---

## Guiding principle

Every AWS service in this plan exists because the project has a problem it
solves. Nothing is here to be named on a resume.

That distinction matters more than it sounds: an interviewer who sees SQS with
no async workload, or a VPC with nothing to isolate, reads it as *adds
complexity without cause* — which is worse than not having it. The architecture
below is small on purpose, and every piece has a "because".

---

## Phase 1 — Finish the API (prerequisite for everything else)

Continue snippet-by-snippet. Remaining pieces:

| snippet | what | why it's ordered here |
|---|---|---|
| 2 | Pydantic models + `GET /api/health` + `POST /api/search` | Non-streaming first — `curl`-verifiable, proves wiring before SSE adds a layer that's hard to debug |
| 3 | `POST /api/chat` with SSE streaming | Generation is seconds; without streaming the page just hangs |
| 4 | `web/index.html` — no build step | Single file, `fetch` + `EventSource` |
| 5 | Static mount + `--port 8001` default | Serve the page from the same process |

**Design decisions already made in snippet 1:**

- Everything loads once in `lifespan`, never per request.
- `_vectors()` is warmed at startup. This is a **correctness** fix, not a
  perf one: the exact-search matrix cache is not thread-safe, and two
  concurrent cold requests would both trigger a full `get()` of every
  embedding. Warming closes the window structurally instead of with a lock.

**Still to decide in snippet 3:** `ollama.chat` is blocking. It has to go
through `AsyncClient` or a threadpool, or one generation stalls the event loop
for every other user. `ollama.AsyncClient` exists — verified.

**The UI must render `dense_rank` / `bm25_rank` / `rrf_score` per chunk.**
The whole design thesis is auditable retrieval, and those fields already exist
on every result. Showing them makes hybrid retrieval *visible* — it is the
single best thing to have on screen when demoing this.

---

## Phase 1.5 — Cross-encoder reranking (B1) — DONE: REJECTED

**Was PLAN-v2 Phase B1, scheduled after the eval expansion. Moved ahead of it
because the relevance gate turned out to have a hard ceiling.**

### Why the reorder

The cosine relevance gate (shipped, `NOTES-changes.md` §10) stops the
demonstrated prompt-injection exploit, but validating it against 19 adversarial
negatives showed it **cannot be tuned any further**:

```
max negative = 0.6231   min positive = 0.4408   separation = -0.1823  OVERLAP
```

`"what learning rate should I use with the Adam optimizer"` scores **0.6231** —
above four genuine positives. It is off-topic but lexically saturated with
corpus vocabulary. Embedding similarity measures *topical proximity*, not
*whether the corpus answers the question*, so no threshold separates them:

| threshold | false-abstain | negatives caught |
|---|---|---|
| 0.37 (current) | 0% | 73% |
| 0.40 | 0% | 82% |
| 0.50 | 13% | 95% |

82% at 0% false-abstain is the ceiling. Everything above it costs real queries.

### What B1 buys, in priority order

1. **A calibrated abstain signal.** A cross-encoder scores query and chunk
   *jointly*, so it can tell "mentions learning rates" from "answers a question
   about learning rates" — the exact distinction embeddings collapse.
2. **The `paraphrase-hallucination` miss** — the one case every retriever mode
   fails. Original justification for B1, still valid.
3. **A second ablation row** (rerank on/off) on top of dense/BM25/hybrid.

Three payoffs from one piece of work; that is what moved it up the queue.

### Approach

- `cross-encoder/ms-marco-MiniLM-L-6-v2`, rerank top-20 → `final_k`.
- Behind `Config.rerank: bool = False` so the ablation can run both ways and
  the change is reversible.
- **Measure the latency cost honestly.** Expect 8 ms → 50-100 ms. Report it as
  a tradeoff table, not a win. It stays <1% of end-to-end once generation is
  included, which is the argument for paying it.
- Re-run `--gate` with cross-encoder scores and compare separation against the
  cosine baseline. **If it does not beat -0.1823, say so and keep cosine.**

### OUTCOME — rejected on measurement (see `NOTES-changes.md` §11)

All three justifications falsified. recall@5 unchanged (93.33%), MRR **worse**
(0.900 → 0.867), abstain AUC **worse** (0.970 → 0.927), the known miss still
missed, at +82 ms/query. Chunk-length truncation (200/80 words) did not rescue
it. Kept behind `Config.rerank = False` so the result is reproducible via
`eval_recall.py --rerank`.

**Most useful finding:** the cross-encoder scored the prompt-injection chunk
**highest of all 22 negatives** (+5.88). It is not wrong — that chunk really is
relevant to "is the capital of France a valid question?" — which means **a
better relevance model is a better injection amplifier**. Relevance and safety
are not the same axis.

### Then come back to the threshold

`Config.min_relevance` stays at 0.37 until B1 lands. Revisit once there is a
better signal to threshold on, and once Phase 2 grows the positive set (n=15
positives is why 0.40 wasn't adopted despite dominating: only 0.04 margin above
the lowest true positive).

---

## Phase 2 — Promote the eval set (do before any deploy work)

62 candidates sit in `evals/REVIEW.md`, drafted from abstracts only (never from
retrieved chunks — that would make the retriever shape its own test set).
Triage: 58 clean · 1 ambiguous · 3 missed.

Promote via `--promote` → **79 cases** (76 scored + 3 negatives). One case then
moves recall ~1.3pp instead of 6.7pp, which is the resolution needed to
actually separate hybrid from BM25.

**Discipline that must hold:** judge a MISS by reading the question and the
paper, *never* by whether retrieval found it. Dropping cases the system fails
strips out exactly what the eval exists to catch and silently inflates recall.
2 of 62 candidates are current failures — if that number were 0, the set would
be worthless.

Then re-run `--ablate`. That run decides whether the headline claim is about
**ranking** (current evidence) or **coverage** (unknown until n is big enough).

---

## Phase 3 — Docker — DONE (built & verified)

**The problem it solves:** running this today means a venv, Ollama, a 9 GB
model pull, and a 190-second ingest. No hiring manager will do that.
`docker compose up` is how the work actually gets evaluated.

- Multi-stage build. Final image must not carry torch build deps.
- **`data/` cannot go in the image** — 107 MB of PDFs plus the Chroma DB. This
  is what forces the S3 decision in Phase 4; it is not optional bloat.
- `docker-compose.yml` with two services: the API, and Ollama for local dev.
  Compose is the right tool for two containers on one host. Kubernetes is not.
- Health check hitting `/api/health` so the container reports honestly.

**Expected friction, worth planning for:** the image will be large because of
torch. Options are CPU-only torch wheels (much smaller) or accepting ~2 GB.
CPU-only is correct here — the deployed service does retrieval, not ingest, so
it never needs MPS/CUDA.

---

## Phase 4 — Deploy, split along the measured cost boundary

**This is the architecture decision, and it comes from our own numbers:**
retrieval is 7.7 ms and CPU-only; generation is seconds and is the only thing
that wants a GPU. So split them.

```
Browser ──► FastAPI retrieval service (CPU, always on, ~$12/mo)
                │
                ├─ exact matmul over 4.4 MB matrix  ──► S3 (index artifacts)
                ├─ BM25
                └─ generation ──► hosted LLM API (per-token, ~$0 idle)
```

**Why not self-host the 14B model:** a GPU instance is roughly $1/hr — about
$700/month, mostly idle, for a portfolio demo. Verify current pricing, but the
order of magnitude is the point.

`generate.py` already has an OpenAI-compatible backend behind
`ARXIV_RAG_BACKEND`, so the seam exists. Ollama stays the local-dev default.

### Compute choice — the reasoning matters more than the pick

| option | verdict |
|---|---|
| **Lambda** | **No.** Cold start + ~90 MB model load + matrix fetch on every scale-to-zero. Wrong shape for a warm in-memory index. |
| **App Runner / Fargate** | **Yes.** Long-lived container, warm caches, scales to a small floor. Matches the lifespan design exactly. |
| **EC2** | Workable, cheapest, but you own patching. |
| **EKS** | **No.** ~$73/mo control plane before nodes, to orchestrate one container. |

Being able to walk this comparison is worth more in an interview than having
deployed any single one of them.

### AWS services, each with a because

| service | because |
|---|---|
| **S3** | 107 MB of PDFs + index can't live in the image, and a stateless container can't rebuild it on boot. Lifecycle rules for old PDFs. |
| **Secrets Manager / SSM** | The hosted-LLM API key must not be baked into an image or a compose file. |
| **IAM task role** | Least-privilege read on one bucket + one secret. Small, real, demonstrable. |
| **CloudWatch** | Log the retrieval mode, latency, and which retriever won per query. Turns the eval story into an operational one. |

---

## Phase 5 — SQS + worker for ingest (optional, but honest)

The one resilience-domain item that isn't theater. Ingest is minutes long and
rate-limited by arXiv's ToS — it genuinely cannot run inside an HTTP request.

```
POST /api/ingest ──► SQS ──► worker (fetch → parse → embed → S3 snapshot)
```

Do this **only after** Phase 4 works. It is a real pattern with a real
justification, which is exactly why it shouldn't be rushed in as decoration.

---

## Phase 6 — Efficiency — DONE (5.6x)

Warm re-index of 115 papers is ~21 s and entirely CPU-bound:

| | wall clock | writes | bytes |
|---|---|---|---|
| baseline | 71.2 s | 115 | 448.1 MB |
| C1 batched BM25 | 58.0 s | 1 | 7.7 MB |
| **C1 + C2** | **12.7 s** | **1** | **7.7 MB** |

**Two estimates in the original plan were wrong and are corrected in
`NOTES-changes.md` §13:** serial parse is 397 ms/PDF (not 121 — I had timed
only `_read_lines`), and warm re-index is 71 s (not 21 — summing
independently-measured parts undercounted by 3.4x). At 1000 papers the O(N²) rebuild would be ~560 s and
~38 GB written — that's the thing that would actually stop this scaling.

**Cold ingest stays network-bound and that is the more impressive fact:** ~90%
of wall clock is arXiv's mandated delay. Not optimizing it — and saying why —
is better judgment than a fake speedup.

---

## Explicit non-goals

- **EKS / Kubernetes.** One container. "Why Kubernetes?" has no good answer
  here except "to learn it", which is a fine reason to build a *separate* toy
  and a weak answer in an interview. Keep this repo's story clean.
- **Multi-AZ / HA.** No uptime requirement. Adding it invites "what's your
  SLA?" and there's no answer.
- **Auto-scaling.** No traffic to scale.
- **RDS / ElastiCache.** No relational data; no cache pressure at 2834 chunks.
- **VPC with private subnets + NAT.** ~$32/mo for the NAT gateway to protect
  nothing in a public demo.
- **Micro-optimizing the 7.7 ms retrieval path.** <1% of user-perceived
  latency once generation is included.

---

## Sequence

| # | phase | effort | gate |
|---|---|---|---|
| 1 | ~~Finish API~~ **DONE** | — | serves on :8001, streaming + gate + web UI |
| 1.5 | ~~Cross-encoder rerank (B1)~~ **DONE — REJECTED** | — | measured worse on every metric; NOTES §11 |
| 2 | Promote eval set, re-ablate | 1 h + review | n=79, conclusion decided |
| 3 | ~~Docker + compose~~ **DONE** | — | 2.62 GB image, 3 s cold start, healthy; NOTES §17 |
| 4 | S3 + deploy + hosted LLM | 1–2 days | live URL |
| 5 | ~~Efficiency (C1/C2)~~ **DONE** | — | 71.2 s → 12.7 s, 448 MB → 7.7 MB |
| 6 | SQS ingest | 1 day | optional |

Phase 2 before Phase 3 deliberately: the eval conclusion is the intellectual
core, and it should not be blocked behind infrastructure work.

---

## On the certifications

**CCP** is foundational and deliberately non-technical — aimed at sales, PM,
and finance staff. For an MS CS grad targeting engineering roles it signals
less than the time costs.

**SAA** is the one worth having, because it tests *choosing between* services
rather than naming them — which is what this plan is made of. Phases 3–5 cover
real ground in all four SAA domains (secure: IAM/Secrets; resilient: SQS
decoupling; performing: compute choice, caching; cost-optimized: the
retrieval/generation split, backed by measurements).

Skills accumulate as consequences of shipping. That version is defensible.
