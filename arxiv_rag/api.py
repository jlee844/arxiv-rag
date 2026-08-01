"""HTTP API over the RAG pipeline.

Run:
    .venv/bin/uvicorn arxiv_rag.api:app --reload --port 8000
"""

from __future__ import annotations



import copy
import re
import time
from typing import Literal

import json

import ollama
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .generate import REFUSAL, _SYSTEM_PROMPT, _build_context

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .retrieve import RETRIEVERS, max_dense_score

from contextlib import asynccontextmanager
from pathlib import Path

from .config import Config
from .embed import embed_query
from .index import PaperIndex

WEB_DIR = Path(__file__).parent.parent / "web"

# Process-wide singletons, built once at startup by lifespan().
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the index and warm every cache BEFORE the first request lands."""
    cfg = Config()
    index = PaperIndex(cfg)

    # Warm two caches that would otherwise be paid per-request or raced:
    #  1. embed.py's @lru_cache holds the SentenceTransformer — ~3s of disk
    #     read + torch init. A cold worker has nothing cached.
    #  2. PaperIndex._vectors() builds the 4.4 MB exact-search matrix lazily,
    #     and it is NOT thread-safe. Two concurrent cold requests would BOTH
    #     see `self._matrix is None` and BOTH run the full get(). Touching it
    #     here means that race window never exists under uvicorn.
    warm = embed_query("warmup", cfg.embed_model, device=cfg.embed_device)
    index.dense_search(warm.tolist(), k=1)

    _state["cfg"] = cfg
    _state["index"] = index
    print(f"ready: {index.count()} chunks / {len(index.indexed_papers())} papers")

    yield                      # ← app serves requests here

    _state.clear()


app = FastAPI(title="arxiv-rag", version="1.0", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int | None = Field(None, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


class Source(BaseModel):
    """One retrieved chunk, as exposed to clients."""
    chunk_id: str
    arxiv_id: str
    title: str
    section: str
    published: str
    text: str
    # Retrieval provenance — makes hybrid retrieval visible.
    rrf_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    took_ms: float
    results: list[Source]


class HealthResponse(BaseModel):
    status: str
    chunks: int
    papers: int
    embed_model: str
    llm_model: str
    exact_search: bool


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg, index = _state["cfg"], _state["index"]
    n = index.count()
    return HealthResponse(
        status="ok" if n else "empty",
        chunks=n,
        papers=len(index.indexed_papers()),
        embed_model=cfg.embed_model,
        llm_model=cfg.ollama_model,
        exact_search=n <= cfg.exact_search_max,
    )


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Retrieval only — no generation. Fast, and the best debugging surface."""
    cfg, index = _state["cfg"], _state["index"]
    if index.count() == 0:
        raise HTTPException(status_code=503, detail="Index is empty. Run ingest.")

    # NEVER mutate the shared cfg — see note 3 below.
    if req.k is not None:
        cfg = copy.copy(cfg)
        cfg.final_k = req.k

    t0 = time.perf_counter()
    hits = RETRIEVERS[req.mode](req.query, index, cfg)
    took = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        count=len(hits),
        took_ms=round(took, 2),
        results=[Source(**h) for h in hits],
    )

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int | None = Field(None, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. The blank line terminates the message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# Citation markers the model is asked to emit: [1], [2], ...
_CITE = re.compile(r"\[(\d{1,2})\]")

@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Retrieve, then stream a grounded answer token by token."""
    cfg, index = _state["cfg"], _state["index"]
    if index.count() == 0:
        raise HTTPException(status_code=503, detail="Index is empty. Run ingest.")
    if req.k is not None:
        cfg = copy.copy(cfg)
        cfg.final_k = req.k

    async def stream():
        # retrieve() is blocking CPU work. In an async handler it MUST go to a
        # threadpool, or it blocks the event loop for every other connection.
        t0 = time.perf_counter()
        hits = await run_in_threadpool(RETRIEVERS[req.mode], req.query, index, cfg)
        took = (time.perf_counter() - t0) * 1000

        # Sources first: the UI renders citations immediately, while the much
        # slower generation is still starting.
        yield _sse("sources", {
            "took_ms": round(took, 2),
            "mode": req.mode,
            "results": [Source(**h).model_dump() for h in hits],
        })

        if not hits:
            yield _sse("done", {"reason": "no_results"})
            return

        # Relevance gate — refuse WITHOUT calling the LLM. See generate.py for
        # why prompt-level defenses are insufficient.
        best = max_dense_score(hits)
        if best < cfg.min_relevance:
            yield _sse("token", {"t": REFUSAL})
            yield _sse("done", {"grounded": True, "cited": [],
                                "abstained": True, "top_similarity": round(best, 4)})
            return

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{_build_context(hits)}\n\nQuestion: {req.query}"},
        ]

        full = ""
        try:
            client = ollama.AsyncClient(host=cfg.ollama_base_url)
            gen = await client.chat(
                model=cfg.ollama_model,
                messages=messages,
                options={"temperature": 0.2},
                stream=True,
            )
            async for part in gen:
                token = part["message"]["content"]
                if token:
                    full += token
                    yield _sse("token", {"t": token})
        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})
            return

        # Deterministic grounding check. An answer citing no valid excerpt was
        # not derived from the context — it came from the model's own weights.
        # This catches parametric leakage ("the capital of France is Paris")
        # WITHOUT an extra model call. It does NOT catch a confident answer
        # citing irrelevant excerpts; that needs relevance gating (PLAN-v2 B2).
        cited = {int(n) for n in _CITE.findall(full)}
        valid = cited & set(range(1, len(hits) + 1))
        yield _sse("done", {"grounded": bool(valid), "cited": sorted(valid)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


from fastapi.staticfiles import StaticFiles

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")