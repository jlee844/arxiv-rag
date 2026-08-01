"""Dual index: ChromaDB (dense) + BM25 (sparse).

STAGE 4 of PLAN-learn.md. Reference: `git show 4c7f66a:arxiv_rag/index.py`

THE CENTRAL DESIGN PROBLEM: ChromaDB persists itself automatically. BM25 does
not — rank_bm25 is a pure in-memory object. Handle this or your index works in
the session that built it and silently returns zero BM25 results forever after.
That's the bug people actually ship.

Design:
  - ChromaDB stores embeddings + metadata, persists to disk on its own.
  - BM25 is pickled, with the corpus written alongside as JSON.
      Why both? The pickle is the scorer; the JSON is the human-readable corpus
      you'll need when retrieval misbehaves. Pickle alone is opaque.
  - Adding papers is idempotent: existing chunk_ids are skipped.
  - Dense search is EXACT (one matmul), not Chroma's HNSW. See
    _dense_search_exact for the measurements that motivated this.
"""

from __future__ import annotations
import contextlib
import json
import pickle
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .config import Config
from .embed import embed_texts
from .parse import Chunk
import logging

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


# Sections whose text is bibliographic rather than substantive. Matched as
# substrings against a whitespace-normalized, lowercased heading, so this
# catches "24\nREFERENCES", "References", "Acknowledgements", etc.
_SKIP_SECTIONS = ("references", "bibliography", "acknowledg")


def _is_noise(chunk: Chunk) -> bool:
    """True if a chunk is bibliography/acknowledgements rather than content.

    DESIGN CHOICE — filtering here (index time) rather than at retrieval time:

      Reference lists are topically dense: a bibliography chunk is a list of
      paper titles, so it has surface overlap with almost ANY query in the
      field. Measured on this corpus, reference chunks took rank 1 for both
      BM25 (score 1.97) and dense (0.362), consuming 2 of 3 top slots with
      citation strings. They crowd out real content.

      Index time wins on: smaller index, cheaper ingest (no embedding spent on
      them), and no per-query cost. Retrieval-time filtering would have to run
      on every search forever, and would need over-fetching (ask for k=20 to
      reliably keep 5) to avoid returning short result lists.

      The cost is real and worth naming: these chunks no longer exist, so
      citation-graph queries ("which papers cite Radford?", "what does this
      paper's related work cover?") become unanswerable. That's an acceptable
      trade for a research-QA tool; it would NOT be acceptable for a
      literature-mapping tool. If that changes, move this to retrieval time
      rather than re-ingesting.
    """
    section = " ".join(chunk.section.lower().split())
    return any(token in section for token in _SKIP_SECTIONS)

def _tokenize(text: str) -> list[str]:
    """Tokenizer for BM25. MUST be identical for corpus and query, or scores
    are computed against a vocabulary the query can never match."""
    return text.lower().split()

class PaperIndex:
    """Manages the ChromaDB collection and BM25 index together."""

    COLLECTION_NAME = "arxiv_papers"
    BM25_FILE = "bm25.pkl"
    CORPUS_FILE = "corpus.json"   # parallel list of {chunk_id, text} for BM25

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()

        self._chroma = chromadb.PersistentClient(
            path=str(self.cfg.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self._matrix: np.ndarray | None = None      # lazy exact-search cache
        self._matrix_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []      # [{chunk_id, text}, ...]
        self._deferred = False   # inside batch(): skip per-call rebuild
        self._dirty = False      # corpus changed since last rebuild
        self._load_bm25()

    @contextlib.contextmanager
    def batch(self):
        """Defer BM25 rebuild+persist until the whole batch is added.

        WHY: BM25Okapi precomputes corpus-wide IDF at construction, so it
        cannot be appended to — every add_chunks() rebuilt the ENTIRE index and
        re-pickled it. Over an ingest that is quadratic in total work.

        Measured on a 106-paper / 2834-chunk ingest:
            per-paper : 6.26 s rebuild CPU, 384.7 MB written
            batched   : 0.11 s rebuild CPU,   7.4 MB written   (55x / 52x)

        Searches inside the batch still work: _ensure_bm25() rebuilds lazily if
        anything is dirty, so correctness never depends on remembering to exit.
        """
        self._deferred = True
        try:
            yield self
        finally:
            self._deferred = False
            if self._dirty:
                self._rebuild_bm25()
                self._save_bm25()
                self._dirty = False

    def _ensure_bm25(self):
        """Rebuild if a deferred batch left the index stale."""
        if self._dirty:
            self._rebuild_bm25()
            self._dirty = False
        

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Add chunks to BOTH indexes. Returns number of new chunks added.
        """
        if not chunks:
            return 0
        
        chunks = [c for c in chunks if not _is_noise(c)]
        if not chunks:
            return 0

        # Idempotency: ask Chroma which ids it already has.
        existing = set(self._col.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Embedding", leave=False, unit="batch"):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(embed_texts(batch, self.cfg.embed_model).tolist())

        self._col.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=all_embeddings,
            documents=[c.text for c in new_chunks],
            metadatas=[{
                "arxiv_id": c.arxiv_id,
                "title": c.title,
                "authors": c.authors,
                "published": c.published,
                "section": c.section,
                "chunk_index": c.chunk_index,
            } for c in new_chunks],
        )

        for c in new_chunks:
            self._corpus.append({"chunk_id": c.chunk_id, "text": c.text})

        self._matrix = None          # invalidate exact-search cache
        self._matrix = None          # invalidate exact-search cache
        self._dirty = True
        if not self._deferred:
            self._rebuild_bm25()
            self._save_bm25()
            self._dirty = False
        return len(new_chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    # ── Dense search ──────────────────────────────────────────────────────
    #
    # Chroma's HNSW is an APPROXIMATE index, and at this corpus size it is
    # strictly worse than exact search on both axes we care about:
    #
    #   |              | latency  | deterministic                     |
    #   |--------------|----------|-----------------------------------|
    #   | Chroma HNSW  | ~1.10 ms | NO — 6 identical processes gave 6 |
    #   |              |          | different result fingerprints     |
    #   | exact matmul | ~0.04 ms | YES — 4/4 identical               |
    #
    # HNSW rebuilds its graph per process with threaded insertion, so the
    # approximation shifts run to run. That moved dense recall@5 across
    # 80.0% / 86.7% / 93.3% on an unchanged corpus — larger than any effect
    # the eval was trying to measure.
    #
    # Embeddings are L2-normalized (see embed.py), so cosine similarity IS the
    # dot product and the whole search is one (N, D) @ (D,) matmul over 4.4 MB.
    # Measured crossover where HNSW starts to win is ~80k vectors (~3000
    # papers), so we keep HNSW as a fallback above that.

    def dense_search(self, query_embedding: list[float], k: int = 10) -> list[dict]:
        """Top-k by cosine similarity. Exact below cfg.exact_search_max."""
        n = self._col.count()
        if n == 0:
            return []                      # Chroma rejects n_results=0
        if n <= getattr(self.cfg, "exact_search_max", 80_000):
            return self._dense_search_exact(query_embedding, k)
        return self._dense_search_hnsw(query_embedding, k)

    def _vectors(self) -> tuple[np.ndarray, list[str]]:
        """Cached (N, D) matrix of all embeddings plus their ids.

        Only the matrix is held (4.4 MB at 2834 chunks, ~123 MB at the 80k
        ceiling). Documents and metadata are fetched per query in one batched
        get, same as bm25_search, so memory stays bounded by the vectors.
        """
        if self._matrix is None:
            got = self._col.get(include=["embeddings"])
            self._matrix_ids = got["ids"]
            self._matrix = (np.asarray(got["embeddings"], dtype=np.float32)
                            if got["ids"] else np.zeros((0, 1), dtype=np.float32))
        return self._matrix, self._matrix_ids

    def _dense_search_exact(self, query_embedding: list[float], k: int) -> list[dict]:
        matrix, ids = self._vectors()
        if not len(ids):
            return []

        scores = matrix @ np.asarray(query_embedding, dtype=np.float32)
        k = min(k, len(ids))
        # argpartition is O(N) vs O(N log N) for a full sort; we only sort the k.
        top = np.argpartition(-scores, k - 1)[:k] if k < len(ids) else np.arange(len(ids))
        top = top[np.argsort(-scores[top], kind="stable")]

        hit_ids = [ids[i] for i in top]
        fetched = self._col.get(ids=hit_ids, include=["metadatas", "documents"])
        by_id = {cid: (fetched["documents"][j], fetched["metadatas"][j])
                 for j, cid in enumerate(fetched["ids"])}

        out = []
        for idx, cid in zip(top, hit_ids):
            if cid not in by_id:
                continue
            doc, meta = by_id[cid]
            out.append({"chunk_id": cid, "text": doc,
                        "score": float(scores[idx]), **meta})
        return out

    def _dense_search_hnsw(self, query_embedding: list[float], k: int) -> list[dict]:
        """Approximate fallback for corpora past the exact-search crossover."""
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [{
            "chunk_id": cid,
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],   # distance -> similarity
            **results["metadatas"][0][i],
        } for i, cid in enumerate(results["ids"][0])]

    def bm25_search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k results from BM25 sparse retrieval.

        Query tokenization must match corpus tokenization (_tokenize).
        Non-positive scores (no term overlap) are dropped.
        """
        if self._bm25 is None or not self._corpus:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits = [(self._corpus[i]["chunk_id"], float(scores[i]))
                for i in ranked if scores[i] > 0]
        if not hits:
            return []

        # ONE batched fetch, not one per hit.
        ids = [cid for cid, _ in hits]
        fetched = self._col.get(ids=ids, include=["metadatas", "documents"])
        by_id = {cid: (fetched["documents"][i], fetched["metadatas"][i])
                 for i, cid in enumerate(fetched["ids"])}

        output = []
        for cid, score in hits:                # preserve BM25 rank order
            if cid not in by_id:
                continue
            doc, meta = by_id[cid]
            output.append({"chunk_id": cid, "text": doc, "score": score, **meta})
        return output

    def count(self) -> int:
        """Number of chunks in the Chroma collection."""
        return self._col.count()

    def indexed_papers(self) -> list[str]:
        """Return the unique arxiv_ids currently indexed."""
        if self._col.count() == 0:
            return []
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return list({m["arxiv_id"] for m in all_meta})

    # ── BM25 persistence ──────────────────────────────────────────────────────

    def _rebuild_bm25(self):
        """Rebuild BM25Okapi from self._corpus. None if empty (BM25Okapi([]) raises)."""
        tokenized = [_tokenize(e["text"]) for e in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _save_bm25(self):
        with open(self.cfg.bm25_dir / self.BM25_FILE, "wb") as f:
            pickle.dump(self._bm25, f)
        with open(self.cfg.bm25_dir / self.CORPUS_FILE, "w") as f:
            json.dump(self._corpus, f)

    def _load_bm25(self):
        bm25_path = self.cfg.bm25_dir / self.BM25_FILE
        corpus_path = self.cfg.bm25_dir / self.CORPUS_FILE
        if bm25_path.exists() and corpus_path.exists():
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(corpus_path) as f:
                self._corpus = json.load(f)
