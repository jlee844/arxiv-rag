"""PDF parsing and chunking.

Strategy:
  - Abstract is always its own chunk (highest-signal 200 words in the paper).
  - Section headings are detected by TYPOGRAPHY, not by regex: real headings are
    bold and set larger than body text. Table rows and figure captions are not.
  - Body splits at those headings first, then into overlapping word windows so a
    3000-word Results section doesn't become one chunk.
  - Each chunk carries full metadata so retrieval results are self-contained.
"""

from __future__ import annotations
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# Fallback only, for PDFs with no usable font metadata (scans, odd producers).
_SECTION_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z][A-Za-z\s]{3,50}|"
    r"Abstract|Introduction|Related Work|Background|"
    r"Method(?:ology|s)?|Approach|"
    r"Experiment(?:s|al Setup)?|Results?|"
    r"Discussion|Conclusions?|Limitations?|"
    r"Appendix|References|Bibliography)\s*$",
    re.MULTILINE,
)

_BOLD_FLAG = 1 << 4          # PyMuPDF span flag bit for bold
_SIZE_TOLERANCE = 0.3        # pt a heading must exceed body size by
_MAX_HEADING_WORDS = 12      # longer than this and it's prose, not a heading


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


def _clean(text: str) -> str:
    """Undo print-typesetting artifacts. ORDER MATTERS."""
    # 1. Ligatures: 'ﬁ' (U+FB01) is ONE char. NFKD decomposes it to f + i.
    text = unicodedata.normalize("NFKD", text)
    # 2. Rejoin words hyphenated across a line break. MUST precede step 3, or
    #    "frac-\ntion" becomes "frac- tion" and is unrecoverable.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 3. Newline + lowercase = soft wrap mid-sentence, not a real break.
    text = re.sub(r"\n(?=[a-z])", " ", text)
    # 4. Collapse blank-line runs.
    return re.sub(r"\n{3,}", "\n\n", text)


def _norm_heading(text: str) -> str:
    """Normalize a heading for use as metadata: ligatures out, whitespace flat.

    Headings need NFKD too — without it you get sections literally named
    "task-speciﬁc vl models", which breaks any equality/filter test on them.
    """
    return " ".join(unicodedata.normalize("NFKD", text).split())


def _read_lines(pdf_path: Path) -> list[tuple[str, float, bool]]:
    """Return [(line_text, font_size, is_bold), ...] in reading order."""
    lines: list[tuple[str, float, bool]] = []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                first = spans[0]
                bold = bool(first["flags"] & _BOLD_FLAG) or "bold" in first["font"].lower()
                lines.append((text, round(first["size"], 1), bold))
    doc.close()
    return lines


def _body_size(lines: list[tuple[str, float, bool]]) -> float:
    """Modal font size of body text, weighted by character count.

    Weighting by characters rather than line count matters: headings are short,
    body paragraphs are long, so character weighting makes body text dominate
    even in a paper with many headings.
    """
    weights: Counter[float] = Counter()
    for text, size, _ in lines:
        weights[size] += len(text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _segment(lines, body_size: float) -> list[tuple[str, str]]:
    """Group lines into [(section_heading, body_text), ...] using typography."""
    segments: list[tuple[str, str]] = []
    current = "body"
    buf: list[str] = []

    for text, size, bold in lines:
        is_heading = (
            bold
            and size > body_size + _SIZE_TOLERANCE
            and _word_count(text) <= _MAX_HEADING_WORDS
        )
        if is_heading:
            if buf:
                segments.append((current, "\n".join(buf)))
                buf = []
            current = text
        else:
            buf.append(text)

    if buf:
        segments.append((current, "\n".join(buf)))
    return segments


def _segment_by_regex(full_text: str) -> list[tuple[str, str]]:
    """Fallback segmentation when a PDF exposes no usable font metadata."""
    segments: list[tuple[str, str]] = []
    current, buf = "body", ""
    for part in _SECTION_RE.split(full_text):
        if part and _SECTION_RE.match(part.strip()):
            if buf:
                segments.append((current, buf))
            current, buf = part.strip(), ""
        else:
            buf += " " + (part or "")
    if buf:
        segments.append((current, buf))
    return segments


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
    lines = _read_lines(pdf_path)
    segments = _segment(lines, _body_size(lines)) if lines else []

    # If typography told us nothing, fall back to pattern matching.
    if len(segments) <= 1:
        segments = _segment_by_regex(_clean("\n".join(t for t, _, _ in lines)))

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

    for section, raw in segments:
        _make_chunk(_norm_heading(section), _clean(raw))

    return chunks