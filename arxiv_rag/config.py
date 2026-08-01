"""Central config — all tuneable knobs in one place."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    # Paths
    data_dir: Path = ROOT / "data"
    pdf_dir: Path = ROOT / "data" / "pdfs"
    chroma_dir: Path = ROOT / "data" / "chroma_db"
    bm25_dir: Path = ROOT / "data" / "bm25"

    # Embedding model — fast CPU model, multilingual-capable
    # swap to "all-mpnet-base-v2" for better quality at ~2x slower
    embed_model: str = "all-MiniLM-L6-v2"
    # None = auto-detect (mps on Apple Silicon). Pin to "cpu" for REPRODUCIBLE
    # measurement: MPS and CPU differ by ~2e-7 per component, which is enough to
    # flip near-tied HNSW neighbours and move recall by a whole case.
    embed_device: str | None = None

    # Chunking
    chunk_size: int = 512       # tokens (approximate, split by words)
    chunk_overlap: int = 64     # token overlap between consecutive chunks

    # Retrieval (hybrid RRF — no score weights; ranks only)
    top_k: int = 8              # candidates from each retriever before fusion
    final_k: int = 5            # chunks passed to the LLM
    exact_search_max: int = 80_000   # above this, fall back to Chroma HNSW.
                                # Measured crossover: exact matmul is ~0.02ms
                                # @2.8k and ~1.1ms @100k; HNSW is ~1.1ms flat.
    # Abstain gate. If the best DENSE cosine similarity is below this, no
    # excerpt is topically close and we refuse instead of generating.
    # Measured separation on this corpus: off-topic max 0.336, on-topic min
    # 0.403 (n=14). 0.37 sits in that gap. RRF scores CANNOT be used here —
    # they fuse ranks and discard magnitude, which is exactly the signal a
    # relevance gate needs (see EVAL.md "Out-of-distribution queries").
    min_relevance: float = 0.37

    # Cross-encoder reranking. DEFAULT OFF — measured WORSE than plain hybrid
    # on this corpus (MRR 0.867 vs 0.900, abstain-AUC 0.927 vs 0.970, +82ms).
    # Kept flag-gated so the negative result stays reproducible. See
    # NOTES-changes.md §11 before turning this on.
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20

    rrf_k: int = 60             # RRF damping. Higher = flatter = rewards
                                # cross-retriever consensus over any single
                                # retriever's confidence. 60 is the value from
                                # the original RRF paper; swept in EVAL.md.

    # Generation backend: "ollama" | "openai"
    # Ollama runs locally with Metal on Apple Silicon
    llm_backend: str = field(
        default_factory=lambda: os.getenv("ARXIV_RAG_BACKEND", "ollama")
    )

    # Ollama settings
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    )
    # Env-configurable: under docker-compose the Ollama service is another
    # container ("http://ollama:11434"), not localhost.
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # OpenAI fallback (set OPENAI_API_KEY)
    openai_model: str = "gpt-4o-mini"

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)
