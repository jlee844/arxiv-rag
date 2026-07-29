"""LLM generation with retrieved context.

STAGE 6 of PLAN-learn.md. Reference: `git show 4c7f66a:arxiv_rag/generate.py`

Backends:
  - "ollama"  -> local model (default llama3.2:3b, no GPU needed on Mac)
  - "openai"  -> OpenAI API (set OPENAI_API_KEY, gpt-4o-mini by default)

Keep the backend branch at the top of generate() and give both the same
signature: one prompt, two transports.
"""

from __future__ import annotations

from .config import Config


# TODO: write the system prompt. It is the safety mechanism of the whole
# system, and it needs exactly three things:
#
#   1. "Answer using ONLY the excerpts provided below"
#        -> scopes the model to your retrieved context.
#   2. "Cite the paper title in brackets, e.g. [PercepTax]"
#        -> makes claims auditable. Without citations you cannot distinguish a
#           grounded answer from a hallucination.
#   3. "If the excerpts don't contain enough information, say so clearly"
#        -> THE MOST IMPORTANT LINE. Without an explicit escape hatch, models
#           pattern-match toward answering anyway. With it, they'll decline.
#
# You can watch #3 work: asked about cross-property reasoning over a corpus
# that had no such benchmark, the reference build said it couldn't find one
# instead of inventing it. That line is the difference between a demo and a tool.
_SYSTEM_PROMPT = None


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt.

    TODO: number them [1], [2], ... so citations are checkable against the
    retrieval output printed above the answer. Include title, published date
    and section in each header, then the text. Separate blocks with a clear
    delimiter (e.g. "\\n\\n---\\n\\n").
    """
    raise NotImplementedError


def generate(query: str, chunks: list[dict], config: Config | None = None,
             stream: bool = True) -> str:
    """Generate an answer grounded in the retrieved chunks.

    Args:
        query: The original user query.
        chunks: Retrieved chunks from retrieve().
        config: Config (uses defaults if None).
        stream: If True, print tokens as they arrive (Ollama only).

    Returns:
        Generated answer string.

    TODO: build the user message as "Context:\\n{context}\\n\\nQuestion: {query}",
    then dispatch on cfg.llm_backend.

    CHECKPOINT: ask something your corpus definitely can't answer ("what is the
    capital of France?"). A correct system REFUSES. If it answers, your system
    prompt is too weak — go back and strengthen rule #3.
    """
    raise NotImplementedError


def _generate_ollama(user_message: str, cfg: Config, stream: bool = True) -> str:
    """Call the local Ollama server.

    TODO: ollama.chat(model=cfg.ollama_model, messages=[system, user], stream=...).
    When streaming, each chunk is chunk["message"]["content"] — print it with
    end="", flush=True and accumulate the full string to return.
    """
    raise NotImplementedError


def _generate_openai(user_message: str, cfg: Config) -> str:
    """Call the OpenAI API (requires OPENAI_API_KEY).

    TODO: read the key from env and raise a clear error if missing. Use
    temperature=0.2 — this is a factual grounding task, not creative writing.
    """
    raise NotImplementedError
