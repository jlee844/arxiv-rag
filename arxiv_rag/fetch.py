"""Fetch papers from arXiv API and download PDFs.

Usage:
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
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str   # ISO date string
    pdf_url: str
    entry_url: str


def fetch_papers(query: str, max_results: int = 50) -> list[Paper]:
    """Search arXiv and return Paper metadata (no download yet).

    Args:
        query: Free-text search query, e.g. "vision language model evaluation benchmark"
        max_results: Number of results to fetch (max 300 per arXiv ToS)
    """
    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=3.0,    # respect arXiv rate limits
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        papers.append(Paper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[a.name for a in result.authors],
            abstract=result.summary.replace("\n", " "),
            categories=result.categories,
            published=result.published.strftime("%Y-%m-%d"),
            pdf_url=result.pdf_url,
            entry_url=result.entry_id,
        ))

    return papers


def download_pdf(paper: Paper, pdf_dir: Path, skip_existing: bool = True) -> Path:
    """Download a paper's PDF to pdf_dir/<arxiv_id>.pdf."""
    pdf_dir = Path(pdf_dir)
    dest = pdf_dir / f"{paper.arxiv_id}.pdf"

    if skip_existing and dest.exists():
        return dest

    # arXiv asks for a delay between requests
    time.sleep(1.5)
    resp = requests.get(paper.pdf_url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def download_all(papers: list[Paper], pdf_dir: Path) -> dict[str, Path]:
    """Download all PDFs; returns {arxiv_id: local_path}."""
    paths = {}
    for paper in tqdm(papers, desc="Downloading PDFs"):
        try:
            paths[paper.arxiv_id] = download_pdf(paper, pdf_dir)
        except Exception as e:
            print(f"  [skip] {paper.arxiv_id}: {e}")
    return paths
