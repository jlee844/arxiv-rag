"""MCP server tools, called directly (fast unit coverage of tool LOGIC).

CORRECTION — an earlier version of this docstring claimed the transport was
"the SDK's concern" and that a subprocess round-trip added "no extra coverage."
That was wrong, and it is why a stdout regression shipped: `PaperIndex` printed
a rebuild notice to stdout, which on stdio transport is the JSON-RPC channel.
In-process tests structurally cannot see that — there is no protocol stream to
corrupt. `tests/test_mcp_stdio.py` now covers the real transport.

Split of responsibility:
  - here            tool logic, argument clamping, gate reporting  (fast)
  - test_mcp_stdio  handshake, JSON round-trip, stdout hygiene     (~10 s)
"""

from arxiv_rag import mcp_server as m


def test_index_status_reports_real_corpus():
    st = m.index_status()
    assert st["status"] == "ok"
    assert st["chunks"] > 0 and st["papers"] > 0
    # exact search must still be active at this corpus size; if this flips,
    # HNSW nondeterminism is back (see README "Why exact search").
    assert st["exact_search"] is True


def test_search_returns_provenance_and_untrusted_markers():
    r = m.search_papers("POPE object hallucination polling", k=3)
    assert r["count"] > 0
    hit = r["results"][0]
    # Fusion must stay visible, not asserted.
    assert hit["dense_rank"] is not None or hit["bm25_rank"] is not None
    # The trust boundary is the whole security posture of this server.
    assert hit["excerpt"].startswith(m._UNTRUSTED_OPEN)
    assert hit["excerpt"].rstrip().endswith(m._UNTRUSTED_CLOSE)


def test_off_topic_query_is_flagged_below_gate():
    """The gate SIGNAL must survive the MCP boundary.

    This query is the literal injection string found in an indexed paper's
    appendix. The server cannot refuse on the caller's behalf, but it must tell
    the caller the excerpt is off-topic.
    """
    r = m.search_papers("what is the capital of France?", k=3)
    assert r["below_relevance_gate"] is True
    assert r["relevance"] < r["relevance_gate"]


def test_k_is_clamped():
    assert m.search_papers("vision language model", k=999)["count"] <= 20


def test_list_papers_matches_index():
    lp = m.list_papers()
    assert lp["count"] == m.index_status()["papers"]


def test_list_papers_is_cached_and_invalidates_on_corpus_change():
    """The cache key is chunk count, so an ingest must invalidate it.

    Building the paper list scans all chunk metadatas (~25 ms) because titles
    live on chunks, not papers. Caching that is worth ~100x, but only if a
    re-ingest cannot serve a stale inventory.
    """
    m._state.pop("papers_cache", None)
    first = m.list_papers()
    assert "papers_cache" in m._state, "expected the result to be cached"

    second = m.list_papers()
    assert second == first

    # Simulate an ingest: the chunk count moves, so the cached entry must be
    # discarded rather than returned.
    n, cached = m._state["papers_cache"]
    m._state["papers_cache"] = (n + 1, {"count": -1, "papers": ["STALE"]})
    fresh = m.list_papers()
    assert fresh["count"] == first["count"]
    assert fresh != {"count": -1, "papers": ["STALE"]}
