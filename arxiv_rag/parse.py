"""PDF parsing and chunking.

Strategy:
  - Abstract is always its own chunk (highest-signal 200 words in the paper).
  - Body splits at detected section boundaries first, then into overlapping
    word windows so a 3000-word Results section doesn't become one chunk.
  - Each chunk carries full metadata so retrieval results are self-contained.
"""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# Section headers in ML papers: numbered ("1. Introduction") or bare ("Results").
# The outer parens are a CAPTURE GROUP so re.split keeps the headings.
_SECTION_RE = re.compile(
    r"^\s*(\d+\.?\s+[A-Z][A-Za-z\s]{3,60}|"
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
    """One retrievable unit of text plus everything needed to cite it."""
    chunk_id: str        # "{arxiv_id}_{idx}" — deterministic, enables dedupe
    arxiv_id: str
    title: str
    authors: str         # comma-joined; Chroma metadata can't hold a list
    published: str
    section: str         # detected heading, or "body"
    text: str
    chunk_index: int


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of ~size words.

    Stride is `size - overlap`: advance 448, emit 512, overlap 64.
    """
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return chunks


def parse_pdf(pdf_path: Path, paper, chunk_size: int = 512,
              chunk_overlap: int = 64) -> list[Chunk]:
    """Parse a PDF into a list of overlapping text chunks with metadata.

    Args:
        pdf_path: Path to the downloaded PDF.
        paper: A fetch.Paper instance (supplies metadata).
        chunk_size: Target chunk size in WORDS (~512 words == ~384 tokens).
        chunk_overlap: Word overlap between consecutive chunks.

    Returns:
        List of Chunk objects ready for embedding.
    """
    # --- Pass A: extract raw text ---
    doc = fitz.open(str(pdf_path))
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    # --- Pass B: clean PDF artifacts (ORDER MATTERS) ---
    # 1. Ligatures: 'ﬁ' (U+FB01) is ONE char. NFKD decomposes it to f + i.
    full_text = unicodedata.normalize("NFKD", full_text)
    # 2. Rejoin words hyphenated across a line break. MUST precede step 3,
    #    or "frac-\ntion" becomes "frac- tion" and is unrecoverable.
    full_text = re.sub(r"(\w)-\n(\w)", r"\1\2", full_text)
    # 3. Newline + lowercase = soft wrap mid-sentence, not a real break.
    full_text = re.sub(r"\n(?=[a-z])", " ", full_text)
    # 4. Collapse blank-line runs.
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    # --- Pass C: chunk ---
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

    # Abstract gets its own chunk — from the API, not scraped from the PDF.
    _make_chunk("Abstract", paper.abstract)

    # Body: split on headings, accumulate text under the current heading.
    parts = _SECTION_RE.split(full_text)
    current_section = "body"
    body_buf = ""

    for part in parts:
        if part and _SECTION_RE.match(part.strip()):
            if body_buf:
                _make_chunk(current_section, body_buf)
            current_section = part.strip()
            body_buf = ""
        else:
            body_buf += " " + (part or "")

    if body_buf:
        _make_chunk(current_section, body_buf)

    return chunks