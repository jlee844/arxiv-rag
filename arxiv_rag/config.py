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

    # Chunking
    chunk_size: int = 512       # tokens (approximate, split by words)
    chunk_overlap: int = 64     # token overlap between consecutive chunks

    # Retrieval
    top_k: int = 8              # candidates from each retriever before fusion
    final_k: int = 5            # chunks passed to the LLM
    dense_weight: float = 0.65  # weight for dense score in fusion (rest goes to BM25)

    # Generation backend: "ollama" | "openai"
    # Ollama runs locally with no GPU on Apple Silicon (Metal) or CPU
    llm_backend: str = field(
        default_factory=lambda: os.getenv("ARXIV_RAG_BACKEND", "ollama")
    )

    # Ollama settings
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    )
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI fallback (set OPENAI_API_KEY)
    openai_model: str = "gpt-4o-mini"

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)
