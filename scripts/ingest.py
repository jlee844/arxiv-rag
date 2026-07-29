#!/usr/bin/env python3
"""Ingest papers into the index.

Examples:
    # Search and ingest 30 papers on VLM evaluation
    python scripts/ingest.py "vision language model evaluation benchmark" --n 30

    # Ingest by specific arXiv IDs (comma-separated)
    python scripts/ingest.py --ids 2406.01234,2405.99999

    # List what's already indexed
    python scripts/ingest.py --list
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from tqdm import tqdm

from arxiv_rag.config import Config
from arxiv_rag.fetch import fetch_papers, download_all
from arxiv_rag.index import PaperIndex
from arxiv_rag.parse import parse_pdf


@click.command()
@click.argument("query", default="", required=False)
@click.option("--n", default=20, show_default=True, help="Number of papers to fetch")
@click.option("--ids", default="", help="Comma-separated arXiv IDs to ingest directly")
@click.option("--list", "list_papers", is_flag=True, help="List indexed papers and exit")
def main(query, n, ids, list_papers):
    """Fetch arXiv papers matching QUERY and add them to the local index."""
    cfg = Config()
    index = PaperIndex(cfg)

    if list_papers:
        indexed = index.indexed_papers()
        if not indexed:
            click.echo("Index is empty. Run: python scripts/ingest.py 'your query'")
        else:
            click.echo(f"\n{len(indexed)} papers indexed:\n")
            for arxiv_id in sorted(indexed):
                click.echo(f"  {arxiv_id}")
        return

    if not query and not ids:
        click.echo("Provide a QUERY or --ids. Use --list to see what's indexed.", err=True)
        raise SystemExit(1)

    # --- Fetch ---
    if ids:
        import arxiv
        client = arxiv.Client()
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
        search = arxiv.Search(id_list=id_list)
        from arxiv_rag.fetch import Paper
        papers = []
        for r in client.results(search):
            papers.append(Paper(
                arxiv_id=r.entry_id.split("/")[-1],
                title=r.title,
                authors=[a.name for a in r.authors],
                abstract=r.summary.replace("\n", " "),
                categories=r.categories,
                published=r.published.strftime("%Y-%m-%d"),
                pdf_url=r.pdf_url,
                entry_url=r.entry_id,
            ))
    else:
        click.echo(f"\nSearching arXiv: '{query}' (n={n}) ...")
        papers = fetch_papers(query, max_results=n)
        click.echo(f"Found {len(papers)} papers.")

    # Skip already indexed
    already = set(index.indexed_papers())
    new_papers = [p for p in papers if p.arxiv_id not in already]
    click.echo(f"{len(new_papers)} new papers to index ({len(papers) - len(new_papers)} already indexed).")

    if not new_papers:
        click.echo("Nothing to do.")
        return

    # --- Download PDFs ---
    click.echo("\nDownloading PDFs ...")
    pdf_paths = download_all(new_papers, cfg.pdf_dir)

    # --- Parse + index ---
    click.echo("\nParsing and indexing ...")
    total_chunks = 0
    for paper in tqdm(new_papers, desc="Papers"):
        pdf_path = pdf_paths.get(paper.arxiv_id)
        if not pdf_path or not pdf_path.exists():
            tqdm.write(f"  [skip] {paper.arxiv_id}: PDF not downloaded")
            continue
        try:
            chunks = parse_pdf(pdf_path, paper,
                               chunk_size=cfg.chunk_size,
                               chunk_overlap=cfg.chunk_overlap)
            added = index.add_chunks(chunks)
            total_chunks += added
            tqdm.write(f"  {paper.arxiv_id}: {added} chunks added")
        except Exception as e:
            tqdm.write(f"  [error] {paper.arxiv_id}: {e}")

    click.echo(f"\nDone. Added {total_chunks} chunks. Index now has {index.count()} total chunks.")


if __name__ == "__main__":
    main()
