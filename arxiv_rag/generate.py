"""LLM generation with retrieved context.

Backends:
  - "ollama"  → local model via Ollama (default: llama3.2:3b, no GPU needed on Mac)
  - "openai"  → OpenAI API (set OPENAI_API_KEY, uses gpt-4o-mini by default)

The system prompt grounds the model strictly in retrieved context to reduce
hallucination. Each answer includes paper citations.
"""

from __future__ import annotations

from .config import Config

_SYSTEM_PROMPT = """\
You are a research assistant specialized in machine learning and AI.
Answer the user's question using ONLY the paper excerpts provided below.
For each claim you make, cite the paper title in brackets, e.g. [PercepTax].
If the excerpts don't contain enough information, say so clearly.
Do NOT make up results, numbers, or claims not in the excerpts.
"""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    sections = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['title']} ({c['published']}) — {c['section']}"
        sections.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(sections)


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
    """
    cfg = config or Config()
    context = _build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    if cfg.llm_backend == "openai":
        return _generate_openai(user_message, cfg)
    else:
        return _generate_ollama(user_message, cfg, stream=stream)


def _generate_ollama(user_message: str, cfg: Config, stream: bool = True) -> str:
    """Call Ollama local server."""
    try:
        import ollama
    except ImportError:
        raise RuntimeError("pip install ollama")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if stream:
        full_response = ""
        for chunk in ollama.chat(
            model=cfg.ollama_model,
            messages=messages,
            stream=True,
        ):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token
        print()  # newline after streaming
        return full_response
    else:
        response = ollama.chat(model=cfg.ollama_model, messages=messages)
        return response["message"]["content"]


def _generate_openai(user_message: str, cfg: Config) -> str:
    """Call OpenAI API (requires OPENAI_API_KEY env var)."""
    import os
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI backend.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
