"""VLM figure descriptions — Phase 2 of multimodal retrieval.

WHAT THIS IS FOR, AND WHY IT IS SEPARATE FROM EXTRACTION:

Phase 1 indexes the author's caption, which is free (already in the text layer),
precise, and measured to lift figure-query recall@5 from 71.4% to 100%. That is
the baseline this module has to beat.

So the question is deliberately narrow: **does a generated description add
anything the caption does not already carry?** Captions are often terse
("Figure 4: Ablation results") and omit what the chart actually shows — axis
ranges, direction of the trend, which method wins. A VLM can recover those. Or
it can hallucinate them, describe the wrong panel, or produce generic filler
that dilutes a precise caption into topical mush.

Both outcomes are plausible, which is why `scripts/eval_figures.py` scores
caption-only against caption+description on the same cases rather than assuming
the richer text wins.

COST: ~660 figures. At a few seconds each on a local 7B VLM this is a
tens-of-minutes to hours run, which is why descriptions are cached on disk by
figure_id and never regenerated.

PROMPT DESIGN: the model is asked for retrieval-relevant facts, not prose. A
description that reads well but shares no vocabulary with a user's query is
worthless to BM25 and near-worthless to a bi-encoder.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_HOST = "http://localhost:11434"

# Asks for content, not commentary. "Describe this image" yields "The image
# shows a graph with lines" — fluent, and useless for retrieval.
PROMPT = """You are labelling a figure from a machine-learning paper so it can be found by search.

Write 2-4 sentences stating ONLY what is visibly present:
- the kind of figure (line plot, bar chart, confusion matrix, architecture diagram, qualitative examples, ...)
- what the axes or components are, including units or ranges if legible
- the named methods, models, or datasets that appear
- the direction of the main trend or the visible comparison

Do not speculate about significance, do not restate the caption, and do not
mention that this is a figure from a paper. If something is illegible, omit it
rather than guessing."""


@dataclass
class Description:
    figure_id: str
    text: str
    model: str
    seconds: float


class VLMUnavailable(RuntimeError):
    """Raised when the Ollama host cannot serve the requested vision model.

    Distinguished from a per-figure failure because it means the whole run
    should stop rather than writing hundreds of empty descriptions.
    """


def check_model(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> None:
    """Fail fast and specifically, before a long run starts.

    A stale Ollama SERVER is the failure this repo actually hit: the CLI was
    0.32.5 while the running desktop app served 0.5.12, and pulling a modern
    vision model returned `412: requires a newer version of Ollama`. The error
    names that case explicitly, because "model not found" would send you
    looking in the wrong place.
    """
    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json()
    except Exception as exc:                              # noqa: BLE001
        raise VLMUnavailable(f"cannot reach Ollama at {host}: {exc}") from exc

    names = {m["name"] for m in tags.get("models", [])}
    if model in names:
        return

    try:
        version = requests.get(f"{host}/api/version", timeout=5).json().get("version", "?")
    except Exception:                                     # noqa: BLE001
        version = "?"

    raise VLMUnavailable(
        f"model {model!r} not available (server version {version}).\n"
        f"  installed: {sorted(names) or 'none'}\n"
        f"  If the server version looks older than your `ollama --version`, the\n"
        f"  Ollama desktop app is serving an old build — quit and reopen it,\n"
        f"  then `ollama pull {model}`."
    )


def describe_figure(
    image_path: Path | str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = 180,
) -> Description:
    """Generate one description. Raises on failure; the caller decides policy."""
    image_path = Path(image_path)
    b64 = base64.b64encode(image_path.read_bytes()).decode()

    t0 = time.perf_counter()
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": PROMPT,
            "images": [b64],
            "stream": False,
            # Deterministic, so a re-run reproduces the corpus exactly rather
            # than quietly changing what is indexed.
            "options": {"temperature": 0.0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = (resp.json().get("response") or "").strip()
    if not text:
        raise RuntimeError(f"empty description for {image_path.name}")

    return Description(
        figure_id=image_path.stem,
        text=" ".join(text.split()),
        model=model,
        seconds=round(time.perf_counter() - t0, 2),
    )


def load_cache(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2) + "\n")
