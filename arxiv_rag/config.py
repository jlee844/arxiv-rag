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

    # Embedding model. all-mpnet-base-v2 was TESTED and is WORSE here:
    # dense-only recall 93.42% -> 90.79%, MRR 0.878 -> 0.824, 3.5x slower per
    # query and 37x slower to build the index. It gains only on abstention
    # (AUC 0.975 -> 0.984). See NOTES-changes.md §14.
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
    # Abstain gate on best DENSE cosine. Below this the LLM is never invoked.
    #
    # Measured at n=76 positives / 22 negatives: distributions OVERLAP
    # (max negative 0.6231 > min positive 0.3572), so no value separates them.
    #   0.35 -> 0/76 false-abstain, 68% of negatives caught   <- chosen
    #   0.37 -> 1/76 false-abstain, 73%
    #   0.40 -> 1/76 false-abstain, 82%
    #
    # Chose zero false-abstain: the lowest-scoring positives are ALL short
    # rare-token queries ("THaMES framework", "AI2-THOR environment with SAM
    # and PPO") where dense is weak and BM25 carries retrieval. Refusing a user
    # who typed a real paper name is worse than leaking one off-topic question,
    # which the citation check partly catches anyway.
    #
    # RRF scores cannot be used here: they fuse ranks and discard magnitude,
    # which is precisely the signal a gate needs. See NOTES-changes.md §10/§15.
    min_relevance: float = 0.35

    # Cross-encoder reranking. DEFAULT OFF — measured WORSE than plain hybrid
    # on this corpus (MRR 0.867 vs 0.900, abstain-AUC 0.927 vs 0.970, +82ms).
    # Kept flag-gated so the negative result stays reproducible. See
    # NOTES-changes.md §11 before turning this on.
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20

    # Query expansion. DEFAULT OFF — adds a full LLM call (~2-7 s) in front of
    # ~8 ms retrieval. See NOTES-changes.md §15 before enabling.
    query_expansion: str | None = None      # None | "multi" | "hyde"

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
