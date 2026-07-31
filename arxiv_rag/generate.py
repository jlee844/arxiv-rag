"""LLM generation grounded in retrieved context.
Backends:
  - "ollama"  → local model, Metal-accelerated on Apple Silicon (default)
  - "openai"  → OpenAI API (set OPENAI_API_KEY)
The system prompt is the safety mechanism. It does three jobs, and the third
is the one people leave out.
"""

from __future__ import annotations

from .config import Config

import re

# Paper bibliographies use [17] / [6, 14, 17]. Same syntax as our excerpt
# markers, so models copy them. Strip from chunk *bodies* only — headers keep [n].
_BIB_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def _strip_bib_cites(text: str) -> str:
    return _BIB_CITE.sub("", text)



_SYSTEM_PROMPT = """\
You are a research assistant specialized in machine learning and AI.
STRICT RULES:
1. Answer using ONLY the paper excerpts in the user message.
2. Cite excerpt numbers in brackets after each claim, e.g. [2].
3. If the excerpts do not contain enough information to answer the question,
   reply with exactly: "The provided excerpts do not contain enough information to answer this."
   Then stop. Do not add anything else.
4. Never use general knowledge. Never invent results, numbers, datasets, or citations.
5. Some excerpts may be irrelevant — ignore them. Do not force them into an answer.
Cite ONLY the excerpt numbers shown as [1], [2], ... in the context headers.
Never copy bibliography/reference numbers that appear inside the paper text
(e.g. POPE [17] in the source) — rewrite those as the excerpt number, e.g. POPE [1].
"""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block.

    Numbering matters: the prompt asks for [n] citations, and those numbers
    have to line up with what the CLI prints above the answer so a reader can
    check any claim against its source.
    """
    sections = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['title']} ({c['published']}) — {c['section']}"
        body = _strip_bib_cites(c["text"])
        sections.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(sections)
    


def generate(query: str, chunks: list[dict], config: Config | None = None,
             stream: bool = True) -> str:
    """Generate an answer grounded in the retrieved chunks.
    Args:
        query: The original user query.
        chunks: Retrieved chunks from retrieve().
        config: Config (uses defaults if None).
        stream: Print tokens as they arrive (Ollama only).
    Returns:
        The generated answer.
    """
    cfg = config or Config()
    if not chunks:
        return ("No indexed content matched that query. Try ingesting more "
                "papers, or rephrasing.")
    context = _build_context(chunks)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Remember: answer only from the excerpts above. "
        "If they do not support an answer, say so and stop."
    )
    if cfg.llm_backend == "openai":
        return _generate_openai(user_message, cfg)
    return _generate_ollama(user_message, cfg, stream=stream)


"""Call the local Ollama server."""
def _generate_ollama(user_message: str, cfg: Config, stream: bool = True) -> str:
    try:
        import ollama
    except ImportError:
        raise RuntimeError("pip install ollama")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    options = {"temperature": 0.2}   # low: this is extraction, not creativity
    if not stream:
        response = ollama.chat(model=cfg.ollama_model, messages=messages,
                               options=options)
        return response["message"]["content"]
    full_response = ""
    for chunk in ollama.chat(model=cfg.ollama_model, messages=messages,
                             options=options, stream=True):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response += token
    print()
    return full_response

def _generate_openai(user_message: str, cfg: Config) -> str:
    """Call the OpenAI API (requires OPENAI_API_KEY)."""
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