"""Dual index: ChromaDB (dense) + BM25 (sparse).

Design:
  - ChromaDB stores embeddings + metadata, persists to disk automatically.
  - BM25 index is serialized to JSON alongside ChromaDB so both survive restarts.
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
        self.cfg = config or Config()

        # ChromaDB — persistent, embedded (no server needed)
        self._chroma = chromadb.PersistentClient(
            path=str(self.cfg.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 — load from disk if exists
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []   # [{chunk_id, text}, ...]
        self._load_bm25()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Add chunks to both indexes. Returns number of new chunks added."""
        # Filter already-indexed chunks
        existing_ids = set(self._col.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return 0

        # Embed in batches
        texts = [c.text for c in new_chunks]
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Embedding", leave=False, unit="batch"):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(embed_texts(batch, self.cfg.embed_model).tolist())

        # Add to ChromaDB
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

        # Add to BM25 corpus
        for c in new_chunks:
            self._corpus.append({"chunk_id": c.chunk_id, "text": c.text})

        self._rebuild_bm25()
        self._save_bm25()
        return len(new_chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def dense_search(self, query_embedding: list[float], k: int = 10) -> list[dict]:
        """Return top-k results from ChromaDB by cosine similarity."""
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for i, chunk_id in enumerate(results["ids"][0]):
            output.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine: distance → similarity
                **results["metadatas"][0][i],
            })
        return output

    def bm25_search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k results from BM25 sparse retrieval."""
        if self._bm25 is None or not self._corpus:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        output = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            entry = self._corpus[idx]
            # Fetch metadata from ChromaDB by chunk_id
            meta_result = self._col.get(ids=[entry["chunk_id"]], include=["metadatas", "documents"])
            if not meta_result["ids"]:
                continue
            meta = meta_result["metadatas"][0]
            output.append({
                "chunk_id": entry["chunk_id"],
                "text": meta_result["documents"][0],
                "score": float(scores[idx]),
                **meta,
            })
        return output

    def count(self) -> int:
        return self._col.count()

    def indexed_papers(self) -> list[str]:
        """Return list of unique arxiv_ids currently in the index."""
        if self._col.count() == 0:
            return []
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return list({m["arxiv_id"] for m in all_meta})

    # ── BM25 persistence ──────────────────────────────────────────────────────

    def _rebuild_bm25(self):
        tokenized = [entry["text"].lower().split() for entry in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _save_bm25(self):
        bm25_path = self.cfg.bm25_dir / self.BM25_FILE
        corpus_path = self.cfg.bm25_dir / self.CORPUS_FILE
        with open(bm25_path, "wb") as f:
            pickle.dump(self._bm25, f)
        with open(corpus_path, "w") as f:
            json.dump(self._corpus, f)

    def _load_bm25(self):
        bm25_path = self.cfg.bm25_dir / self.BM25_FILE
        corpus_path = self.cfg.bm25_dir / self.CORPUS_FILE
        if bm25_path.exists() and corpus_path.exists():
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(corpus_path) as f:
                self._corpus = json.load(f)
