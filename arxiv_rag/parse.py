"""PDF parsing and chunking.

Strategy:
  - Abstract is always its own chunk (high signal for overview queries).
  - Body text is split at detected section boundaries first, then by
    token-count windows with overlap so long sections don't become one giant chunk.
  - Each chunk carries full metadata so retrieval results are self-contained.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# Section header patterns common in ML papers
_SECTION_RE = re.compile(
    r"^\s*(\d+\.?\s+[A-Z][A-Za-z\s]{3,60}|"   # "1. Introduction" / "2 Related Work"
    r"Abstract|Introduction|"
    r"Related Work|Background|"
    r"Method(?:ology)?|Approach|Model|"
    r"Experiment(?:s|al Setup)?|Results?|"
    r"Discussion|Conclusion|"
    r"Appendix|References)\s*$",
    re.MULTILINE,
)


@dataclass
class Chunk:
    chunk_id: str        # "{arxiv_id}_{idx}"
    arxiv_id: str
    title: str
    authors: str         # comma-joined
    published: str
    section: str         # detected section heading or "body"
    text: str
    chunk_index: int


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of ~size words."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def parse_pdf(pdf_path: Path, paper, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Parse a PDF into a list of overlapping text chunks with metadata.

    Args:
        pdf_path: Path to the downloaded PDF.
        paper: A fetch.Paper instance (supplies metadata).
        chunk_size: Target chunk size in words (~0.75 tokens/word → ~384 tokens).
        chunk_overlap: Word overlap between consecutive chunks.

    Returns:
        List of Chunk objects ready for embedding.
    """
    doc = fitz.open(str(pdf_path))
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Clean up common PDF artifacts
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)       # collapse blank lines
    full_text = re.sub(r"(\w)-\n(\w)", r"\1\2", full_text)  # rejoin hyphenated words
    full_text = re.sub(r"\n(?=[a-z])", " ", full_text)       # join wrapped lines

    chunks: list[Chunk] = []
    chunk_idx = 0
    authors_str = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += " et al."

    def _make_chunk(section: str, text: str) -> None:
        nonlocal chunk_idx
        text = text.strip()
        if not text or _word_count(text) < 20:
            return
        # Split long sections into overlapping windows
        if _word_count(text) > chunk_size:
            windows = _split_by_words(text, chunk_size, chunk_overlap)
        else:
            windows = [text]

        for window in windows:
            chunks.append(Chunk(
                chunk_id=f"{paper.arxiv_id}_{chunk_idx}",
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=authors_str,
                published=paper.published,
                section=section,
                text=window,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

    # Abstract as dedicated chunk
    abstract_match = re.search(
        r"Abstract[.\s]*\n(.*?)(?=\n\s*\n\s*(?:1[\.\s]|Introduction))",
        full_text, re.DOTALL | re.IGNORECASE
    )
    if abstract_match:
        _make_chunk("Abstract", paper.abstract)  # use arXiv abstract (cleaner than PDF)
    else:
        _make_chunk("Abstract", paper.abstract)

    # Split body by section boundaries
    parts = _SECTION_RE.split(full_text)
    current_section = "body"
    body_buf = ""

    for part in parts:
        if _SECTION_RE.match(part.strip()):
            if body_buf:
                _make_chunk(current_section, body_buf)
            current_section = part.strip()
            body_buf = ""
        else:
            body_buf += " " + part

    if body_buf:
        _make_chunk(current_section, body_buf)

    return chunks
