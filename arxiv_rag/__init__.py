"""arxiv-rag: RAG over arXiv ML/AI papers, runs fully on CPU."""

from .config import Config
from .fetch import fetch_papers
from .parse import parse_pdf
from .index import PaperIndex
from .retrieve import retrieve
from .generate import generate

__all__ = ["Config", "fetch_papers", "parse_pdf", "PaperIndex", "retrieve", "generate"]
