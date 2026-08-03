"""Tests for the BM25 tokenizer and its staleness guard.

The guard is the important half. Changing `_tokenize` without rebuilding the
persisted index does not raise — it silently scores new-vocabulary queries
against old-vocabulary postings, BM25 returns almost nothing, RRF degrades to
dense-only, and the system keeps answering slightly worse forever. These tests
exist so that failure mode cannot come back quietly.
"""

from __future__ import annotations

import json
import pickle

import pytest

from arxiv_rag.index import (
    _TOKENIZER_VERSION,
    PaperIndex,
    _tokenize,
)


class TestTokenize:
    def test_stems_morphological_variants_together(self):
        """The measured win: a query's plural must match the paper's singular."""
        assert _tokenize("hallucinations") == _tokenize("hallucination")
        assert _tokenize("evaluating") == _tokenize("evaluation")

    def test_splits_on_punctuation(self):
        """`.split()` left 'GRPO,' and 'GRPO' as different tokens."""
        assert _tokenize("GRPO, LoRA.") == _tokenize("grpo lora")

    def test_drops_stopwords(self):
        assert "in" not in _tokenize("in the model")
        assert _tokenize("the model") == _tokenize("model")

    def test_drops_single_characters(self):
        """Single chars carry no retrieval signal and bloat the vocabulary."""
        assert _tokenize("a b model") == _tokenize("model")

    def test_is_deterministic(self):
        """Corpus and query tokenization must agree exactly, every call."""
        text = "Evaluating hallucinations in multimodal models (POPE)."
        assert _tokenize(text) == _tokenize(text)

    def test_query_and_corpus_agree(self):
        """The invariant the whole index depends on."""
        corpus = _tokenize("We evaluate object hallucination using POPE.")
        query = _tokenize("evaluating object hallucinations")
        assert set(query) & set(corpus), "query shares no tokens with corpus"

    def test_empty_input(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


class TestStaleIndexGuard:
    """A pickle built with an older tokenizer must be rebuilt, not trusted."""

    def _seed(self, tmp_path, stamp: str | None):
        """Write a corpus + a bm25 pickle, optionally with a tokenizer stamp."""
        d = tmp_path / "bm25"
        d.mkdir(parents=True, exist_ok=True)
        corpus = [{"chunk_id": "c1", "text": "object hallucination evaluation"},
                  {"chunk_id": "c2", "text": "retrieval augmented generation"}]
        (d / PaperIndex.CORPUS_FILE).write_text(json.dumps(corpus))
        # A deliberately WRONG object: if the guard fails to rebuild, whatever
        # loads this will not be a usable BM25 index.
        with open(d / PaperIndex.BM25_FILE, "wb") as f:
            pickle.dump("STALE-SENTINEL", f)
        if stamp is not None:
            (d / PaperIndex.TOKENIZER_FILE).write_text(stamp)
        return d

    def _index(self, tmp_path, bm25_dir):
        from arxiv_rag.config import Config

        cfg = Config()
        cfg.bm25_dir = bm25_dir
        cfg.chroma_dir = tmp_path / "chroma"
        cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
        return PaperIndex(cfg)

    def test_missing_stamp_triggers_rebuild(self, tmp_path):
        """Indexes built before the stamp existed are pre-stemming by definition."""
        d = self._seed(tmp_path, stamp=None)
        idx = self._index(tmp_path, d)
        assert idx._bm25 != "STALE-SENTINEL"
        assert (d / PaperIndex.TOKENIZER_FILE).read_text().strip() == _TOKENIZER_VERSION

    def test_old_stamp_triggers_rebuild(self, tmp_path):
        d = self._seed(tmp_path, stamp="1-split")
        idx = self._index(tmp_path, d)
        assert idx._bm25 != "STALE-SENTINEL"

    def test_current_stamp_loads_without_rebuild(self, tmp_path):
        """The guard must not rebuild on every startup — that would be a 10 s tax."""
        d = self._seed(tmp_path, stamp=_TOKENIZER_VERSION)
        idx = self._index(tmp_path, d)
        assert idx._bm25 == "STALE-SENTINEL", (
            "expected the persisted pickle to be loaded verbatim when the "
            "tokenizer fingerprint matches"
        )

    def test_rebuild_is_persisted(self, tmp_path):
        """A rebuild must write the new stamp, or every startup rebuilds."""
        d = self._seed(tmp_path, stamp="1-split")
        self._index(tmp_path, d)
        assert (d / PaperIndex.TOKENIZER_FILE).read_text().strip() == _TOKENIZER_VERSION
        # Second construction should now take the fast path.
        idx2 = self._index(tmp_path, d)
        assert idx2._bm25 is not None



class TestNoStdoutPollution:
    """Library code must never write to stdout.

    `arxiv_rag.mcp_server` speaks JSON-RPC over stdout. A diagnostic printed
    there is parsed as a protocol frame — the real failure was:

        Invalid JSON: expected value at line 1 column 2
        input_value='[index] BM25 tokenizer changed ... rebuilding over 3288 chunks'

    and it fires on EVERY fresh clone, because a new checkout has no stamp file
    and therefore always rebuilds on first load. Caught only by forcing the
    rebuild path; the happy path is silent and hides it.
    """

    def test_tokenizer_rebuild_is_silent_on_stdout(self, tmp_path, capsys):
        from arxiv_rag.config import Config

        d = tmp_path / "bm25"
        d.mkdir(parents=True)
        corpus = [{"chunk_id": "c1", "text": "object hallucination evaluation"}]
        (d / PaperIndex.CORPUS_FILE).write_text(json.dumps(corpus))
        with open(d / PaperIndex.BM25_FILE, "wb") as f:
            pickle.dump("STALE", f)
        (d / PaperIndex.TOKENIZER_FILE).write_text("1-split")   # forces rebuild

        cfg = Config()
        cfg.bm25_dir = d
        cfg.chroma_dir = tmp_path / "chroma"
        cfg.chroma_dir.mkdir(parents=True)
        PaperIndex(cfg)

        captured = capsys.readouterr()
        assert captured.out == "", (
            f"PaperIndex wrote to stdout, which corrupts the MCP stdio "
            f"transport: {captured.out!r}"
        )
        # The message must still be emitted — just on the correct stream.
        assert "tokenizer changed" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
