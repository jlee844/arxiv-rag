"""PDF parsing and chunking.

STAGE 2 of PLAN-learn.md — the hardest stage, and the one that most determines
answer quality. Reference: `git show 4c7f66a:arxiv_rag/parse.py`

Build this in three passes. Do NOT try to write it in one go:
  Pass A: get raw text out of the PDF and LOOK at it. It will be ugly.
  Pass B: clean the artifacts (see the regex hints below).
  Pass C: chunk with structure — sections first, then word windows.

Target strategy:
  - Abstract is always its own chunk (highest-signal 200 words in the paper).
  - Body splits at detected section boundaries first, then into overlapping
    word windows so a 3000-word Results section doesn't become one chunk.
  - Each chunk carries full metadata so retrieval results are self-contained.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# TODO: write a regex matching section headers in ML papers.
# It needs to catch BOTH numbered ("1. Introduction", "2 Related Work") and
# bare ("Abstract", "Related Work", "Method", "Results", "References") forms.
# Use re.MULTILINE and anchor with ^\s* ... \s*$.
_SECTION_RE = None


@dataclass
class Chunk:
    """One retrievable unit of text plus everything needed to cite it.

    TODO: declare the fields.
      chunk_id ("{arxiv_id}_{idx}"), arxiv_id, title, authors (comma-joined str),
      published, section (detected heading or "body"), text, chunk_index
    """


def _word_count(text: str) -> int:
    """TODO: one line."""
    raise NotImplementedError


def _split_by_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of ~size words.

    WHY overlap: without it, a sentence spanning a chunk boundary is destroyed —
    half its meaning in each chunk, retrievable by neither. 64/512 = 12.5%
    overlap means any sentence survives intact in at least one chunk.

    TODO: split to words, walk with stride (size - overlap), join each window.

    Watch the stride. `i += size` loses the overlap; `i += overlap` is an
    infinite-ish loop producing enormous duplication. It is `size - overlap`.

    tests/test_retrieve.py already tests this function — make those pass.
    """
    raise NotImplementedError


def parse_pdf(pdf_path: Path, paper, chunk_size: int = 512,
              chunk_overlap: int = 64) -> list[Chunk]:
    """Parse a PDF into a list of overlapping text chunks with metadata.

    Args:
        pdf_path: Path to the downloaded PDF.
        paper: A fetch.Paper instance (supplies metadata).
        chunk_size: Target chunk size in WORDS (~0.75 words/token, so
                    512 words ~= 384 tokens — safely under MiniLM's 512 limit).
                    Words are free to count; tokens need the tokenizer loaded.
        chunk_overlap: Word overlap between consecutive chunks.

    TODO — Pass A: extract text
      fitz.open(str(pdf_path)), loop pages, concatenate page.get_text(), close.

    TODO — Pass B: clean PDF artifacts. Three regexes get you most of the way:
      re.sub(r"\\n{3,}", "\\n\\n", t)          collapse blank-line runs
      re.sub(r"(\\w)-\\n(\\w)", r"\\1\\2", t)    rejoin words hyphenated across lines
      re.sub(r"\\n(?=[a-z])", " ", t)          newline + lowercase = soft wrap, not
                                             a new line. A heuristic. Understand
                                             why it's an acceptable one.

    TODO — Pass C: chunk
      1. Emit the abstract as its own chunk. Use paper.abstract (clean, from the
         API), NOT the abstract scraped out of the PDF (column artifacts,
         footnote markers).
      2. Split the body on _SECTION_RE. Track the current section heading.
      3. For each section: if it's longer than chunk_size, run it through
         _split_by_words; otherwise keep it whole.
      4. Drop fragments under ~20 words — page numbers, stray headers, noise.

    Gotcha: re.split() with capture groups returns the DELIMITERS interleaved
    into the result list. That's why you must re-test each part with
    _SECTION_RE.match to decide "heading or content?". Miss this and your
    sections are silently wrong.
    """
    raise NotImplementedError
