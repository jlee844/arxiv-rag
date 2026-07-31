"""Basic smoke tests — no network, no GPU required."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from arxiv_rag.parse import _split_by_words, Chunk
from arxiv_rag.retrieve import _RRF_K


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_split_by_words_basic():
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = _split_by_words(text, size=30, overlap=5)
    assert len(chunks) > 1
    # Each chunk should have at most 30 words
    for c in chunks:
        assert len(c.split()) <= 30


def test_split_by_words_overlap():
    text = " ".join([f"w{i}" for i in range(20)])
    chunks = _split_by_words(text, size=10, overlap=3)
    # First chunk ends at w9; second starts at w7 (10-3=7)
    first_end = chunks[0].split()[-1]
    second_start = chunks[1].split()[0]
    # They should overlap
    assert first_end in chunks[1]


def test_split_short_text_no_split():
    text = "hello world this is a short sentence"
    chunks = _split_by_words(text, size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == text


# ── RRF ───────────────────────────────────────────────────────────────────────

def test_rrf_constant():
    """RRF_K should be positive (standard is 60)."""
    assert _RRF_K > 0


def test_rrf_score_formula():
    """Higher rank (lower index) should yield higher RRF score."""
    scores = [1.0 / (_RRF_K + rank + 1) for rank in range(5)]
    assert scores == sorted(scores, reverse=True)


# ── Config ────────────────────────────────────────────────────────────────────

def test_config_defaults():
    from arxiv_rag.config import Config
    cfg = Config()
    assert cfg.chunk_size > 0
    assert cfg.chunk_overlap < cfg.chunk_size
    assert cfg.final_k <= cfg.top_k
    assert not hasattr(cfg, "dense_weight")  # removed: RRF has no weight term


def test_strip_bib_cites():
    from arxiv_rag.generate import _strip_bib_cites
    text = "POPE [17] and refs [6, 14, 17, 28] evaluate hallucination."
    assert _strip_bib_cites(text) == (
        "POPE  and refs  evaluate hallucination."
    )
