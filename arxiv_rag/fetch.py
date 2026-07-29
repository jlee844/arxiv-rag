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
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str
    pdf_url: str
    entry_url: str




def fetch_papers(query: str, max_results: int = 50) -> list[Paper]:
    """Search arXiv and return Paper metadata (no download yet).

    Args:
        query: Free-text search, e.g. "vision language model evaluation benchmark"
        max_results: Number of results (arXiv ToS caps at 300)
    """
    client = arxiv.Client(
      page_size = min(max_results, 100),
      delay_seconds = 3.0,
      num_retries = 3
    )

    search = arxiv.Search(
      query = query,
      sort_by = arxiv.SortCriterion.Relevance,
      max_results = max_results
    )

    papers = []
    for result in client.results(search):
      papers.append(Paper(
        arxiv_id = result.entry_id.split("/")[-1],
        title = result.title,
        authors = [author.name for author in result.authors],
        abstract = result.summary.replace("\n", " "),
        categories = result.categories,
        published = result.published.strftime("%Y-%m-%d"),
        pdf_url = result.pdf_url,
        entry_url = result.entry_id,
      ))
    return papers

def download_pdf(paper: Paper, pdf_dir: Path, skip_existing: bool = True) -> Path:
    """Download a paper's PDF to pdf_dir/<arxiv_id>.pdf, returning the path.

    TODO:
      1. If skip_existing and the file is already there, return early.
         (You will re-run ingest dozens of times debugging the parser.)
      2. time.sleep(~1.5) before the GET — arXiv is a nonprofit serving free
         full text, and they ask for it.
      3. requests.get(timeout=30), raise_for_status(), write_bytes().
    """
    pdf_dir = Path(pdf_dir)
    dest = pdf_dir / f"{paper.arxiv_id}.pdf"

    if skip_existing and dest.exists():
        return dest

    time.sleep(1.5)                      # arXiv asks for a gap between downloads
    resp = requests.get(paper.pdf_url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def download_all(papers: list[Paper], pdf_dir: Path) -> dict[str, Path]:
    """Download all PDFs; returns {arxiv_id: local_path}.

    TODO: loop with tqdm. Catch exceptions PER PAPER and continue — one
    malformed PDF must not kill a 50-paper ingest.
    """
    paths = {}
    for paper in tqdm(papers, desc="Downloading PDFs"):
        try:
            paths[paper.arxiv_id] = download_pdf(paper, pdf_dir)
        except Exception as e:
            print(f"  [skip] {paper.arxiv_id}: {e}")
    return paths
