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
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .config import Config
from .embed import embed_texts
from .parse import Chunk


class PaperIndex:
    """Manages the ChromaDB collection and BM25 index together."""

    COLLECTION_NAME = "arxiv_papers"
    BM25_FILE = "bm25.pkl"
    CORPUS_FILE = "corpus.json"   # parallel list of {chunk_id, text} for BM25

    def __init__(self, config: Config | None = None):
        """TODO:
          1. self.cfg = config or Config()
          2. chromadb.PersistentClient(path=str(cfg.chroma_dir),
                 settings=Settings(anonymized_telemetry=False))
          3. get_or_create_collection(name=COLLECTION_NAME,
                 metadata={"hnsw:space": "cosine"})   <- must match the
                 normalized embeddings from Stage 3
          4. self._bm25 = None; self._corpus = []; then self._load_bm25()
        """
        raise NotImplementedError

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Add chunks to BOTH indexes. Returns number of new chunks added.

        TODO:
          1. IDEMPOTENCE FIRST: self._col.get(ids=[...])["ids"] gives you what's
             already there. Filter those out. Return 0 if nothing is new.
             Skip this and re-running ingest duplicates chunks, which corrupts
             BM25's document-frequency stats and quietly degrades ranking.
          2. Embed the survivors in batches of batch_size (tqdm is nice here).
          3. self._col.add(ids=, embeddings=, documents=, metadatas=).
             Metadata: arxiv_id, title, authors, published, section, chunk_index.
             Chroma metadata values must be str/int/float/bool — no lists.
          4. Append {chunk_id, text} to self._corpus, then _rebuild_bm25()
             and _save_bm25().
        """
        raise NotImplementedError

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def dense_search(self, query_embedding: list[float], k: int = 10) -> list[dict]:
        """Return top-k results from ChromaDB by cosine similarity.

        CONTRACT — both search methods return the SAME dict shape:
            {chunk_id, text, score, **metadata}
        The fusion layer in Stage 5 must not know which retriever produced a
        result. Get this right and Stage 5 is 40 easy lines.

        TODO: self._col.query(query_embeddings=[...], n_results=...,
              include=["documents", "metadatas", "distances"]).
        Chroma returns lists-of-lists (one per query) — you want index [0].
        Convert distance to similarity: score = 1 - distance.

        Gotcha (this is reference bug #2, don't reproduce it): n_results must
        be >= 1. On an empty collection min(k, count) is 0 and Chroma raises.
        Early-return [] when the collection is empty.
        """
        raise NotImplementedError

    def bm25_search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k results from BM25 sparse retrieval.

        TODO:
          1. Return [] if no index/corpus yet.
          2. Tokenize the query THE SAME WAY you tokenized the corpus
             (.lower().split()). Mismatched tokenization = meaningless scores,
             and it fails silently. This is the #1 cause of "hybrid isn't
             beating dense-only".
          3. self._bm25.get_scores(tokens), take the top k indices.
          4. Drop non-positive scores (no term overlap at all).
          5. Look up metadata by chunk_id and return the same dict shape as
             dense_search.
        """
        raise NotImplementedError

    def count(self) -> int:
        """TODO: number of chunks in the collection."""
        raise NotImplementedError

    def indexed_papers(self) -> list[str]:
        """Return the unique arxiv_ids currently indexed.

        TODO: get(include=["metadatas"]), set-comprehend over "arxiv_id".
        Guard the empty case.
        """
        raise NotImplementedError

    # ── BM25 persistence ──────────────────────────────────────────────────────

    def _rebuild_bm25(self):
        """TODO: tokenize every corpus entry, BM25Okapi(tokenized).
        Set to None if the corpus is empty — BM25Okapi([]) raises."""
        raise NotImplementedError

    def _save_bm25(self):
        """TODO: pickle.dump the bm25 object; json.dump the corpus."""
        raise NotImplementedError

    def _load_bm25(self):
        """TODO: if both files exist, load them back into
        self._bm25 / self._corpus.

        CHECKPOINT for this stage: run a bm25_search in one process, then again
        in a FRESH process. Both must return results. That's the whole point.
        """
        raise NotImplementedError
