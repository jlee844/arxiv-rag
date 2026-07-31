#!/usr/bin/env python3
"""Query the indexed papers.

Examples:
    # Interactive mode (default)
    python scripts/query.py

    # One-shot query
    python scripts/query.py "what methods reduce hallucination in VLMs?"

    # Show retrieved chunks without generating an answer
    python scripts/query.py "LoRA fine-tuning for VLA" --chunks-only

    # Use OpenAI instead of Ollama
    ARXIV_RAG_BACKEND=openai python scripts/query.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from arxiv_rag.config import Config
from arxiv_rag.index import PaperIndex
from arxiv_rag.retrieve import retrieve
from arxiv_rag.generate import generate


def _print_chunks(chunks: list[dict]) -> None:
    click.echo("\n" + "─" * 60)
    click.echo(f"Retrieved {len(chunks)} chunks:\n")
    for i, c in enumerate(chunks, 1):
        click.echo(f"[{i}] {c['title']} ({c['published']}) — {c['section']}")
        click.echo(f"    arXiv:{c['arxiv_id']}  |  RRF score: {c.get('rrf_score', '?')}")
        preview = c["text"][:200].replace("\n", " ")
        click.echo(f"    {preview}...")
        click.echo()


def _run_query(query: str, index: PaperIndex, cfg: Config, chunks_only: bool) -> None:
    if index.count() == 0:
        click.echo("Index is empty. Run: python scripts/ingest.py 'your query'", err=True)
        return

    chunks = retrieve(query, index, cfg)
    if not chunks:
        click.echo("No relevant chunks found. Try ingesting more papers.")
        return

    _print_chunks(chunks)

    if chunks_only:
        return

    click.echo("─" * 60)
    click.echo(f"Answer ({cfg.llm_backend} / {cfg.ollama_model if cfg.llm_backend == 'ollama' else cfg.openai_model}):\n")
    try:
        generate(query, chunks, cfg, stream=(cfg.llm_backend == "ollama"))
    except RuntimeError as e:
        click.echo(f"Generation error: {e}", err=True)
        click.echo("Tip: install Ollama (https://ollama.ai) and run: ollama pull qwen2.5:14b")
    click.echo("─" * 60)


@click.command()
@click.argument("query", default="", required=False)
@click.option("--chunks-only", is_flag=True, help="Show retrieved chunks without generating")
def main(query, chunks_only):
    """Query indexed papers with natural language."""
    cfg = Config()
    index = PaperIndex(cfg)

    if query:
        _run_query(query, index, cfg, chunks_only)
    else:
        # Interactive REPL
        click.echo("\narXiv RAG  |  type your question, 'quit' to exit, '\\chunks' to toggle chunks-only\n")
        chunks_mode = chunks_only
        while True:
            try:
                q = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            if q == r"\chunks":
                chunks_mode = not chunks_mode
                click.echo(f"Chunks-only mode: {'on' if chunks_mode else 'off'}")
                continue
            _run_query(q, index, cfg, chunks_mode)


if __name__ == "__main__":
    main()
