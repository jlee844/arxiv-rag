"""Fetch papers from arXiv API and download PDFs.

STAGE 1 of PLAN-learn.md. Reference: `git show 4c7f66a:arxiv_rag/fetch.py`

Usage (once implemented):
    papers = fetch_papers("VLM hallucination benchmark", max_results=20)
    for p in papers:
        download_pdf(p, pdf_dir)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

import arxiv
import requests
from tqdm import tqdm


@dataclass
class Paper:
    """Our own boundary type over the arxiv library's Result object.

    Why this exists: every downstream module depends on `Paper`, and none of
    them import `arxiv`. Swapping to Semantic Scholar later = rewrite one file.

    TODO: declare the fields.
      arxiv_id, title, authors (list[str]), abstract,
      categories (list[str]), published (ISO date str), pdf_url, entry_url
    """


def fetch_papers(query: str, max_results: int = 50) -> list[Paper]:
    """Search arXiv and return Paper metadata (no download yet).

    Args:
        query: Free-text search, e.g. "vision language model evaluation benchmark"
        max_results: Number of results (arXiv ToS caps at 300)

    TODO:
      1. Build an arxiv.Client. Set delay_seconds=3.0 (ToS) and num_retries=3.
      2. Build an arxiv.Search with sort_by=arxiv.SortCriterion.Relevance.
      3. Map each result -> Paper.

    Gotcha: result.entry_id is a full URL. arxiv_id needs .split("/")[-1],
    or you end up writing PDFs to a filename of "http:".

    Gotcha: result.summary has hard line breaks. Strip them with .replace("\\n", " ").
    """
    raise NotImplementedError


def download_pdf(paper: Paper, pdf_dir: Path, skip_existing: bool = True) -> Path:
    """Download a paper's PDF to pdf_dir/<arxiv_id>.pdf, returning the path.

    TODO:
      1. If skip_existing and the file is already there, return early.
         (You will re-run ingest dozens of times debugging the parser.)
      2. time.sleep(~1.5) before the GET — arXiv is a nonprofit serving free
         full text, and they ask for it.
      3. requests.get(timeout=30), raise_for_status(), write_bytes().
    """
    raise NotImplementedError


def download_all(papers: list[Paper], pdf_dir: Path) -> dict[str, Path]:
    """Download all PDFs; returns {arxiv_id: local_path}.

    TODO: loop with tqdm. Catch exceptions PER PAPER and continue — one
    malformed PDF must not kill a 50-paper ingest.
    """
    raise NotImplementedError
