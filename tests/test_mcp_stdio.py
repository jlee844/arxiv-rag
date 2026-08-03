"""End-to-end MCP test over the REAL stdio transport, in a subprocess.

WHY THIS EXISTS SEPARATELY FROM test_mcp_server.py:

Those tests call the tool functions directly, in-process. That verifies the
tools' logic and nothing about the protocol — and it is precisely why a stdout
regression shipped: `PaperIndex` printed a rebuild notice to stdout, which on
stdio transport IS the JSON-RPC channel, producing

    Invalid JSON: expected value at line 1 column 2
    input_value='[index] BM25 tokenizer changed ... rebuilding over 3288 chunks'

An in-process test cannot see that class of bug, because there is no protocol
stream to corrupt. This one spawns the server exactly as a client would.

These tests are skipped when the index is empty, since they assert against the
real corpus.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

pytest.importorskip("mcp", reason="mcp SDK not installed")


def _index_ready() -> bool:
    try:
        from arxiv_rag.config import Config
        from arxiv_rag.index import PaperIndex

        return PaperIndex(Config()).count() > 0
    except Exception:                                    # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _index_ready(), reason="no index; run scripts/ingest.py"
)

# Model load + index warm on first tool call. Generous, because a timeout here
# would be a flaky failure rather than a real one.
TIMEOUT = 180


async def _session(fn):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "arxiv_rag.mcp_server"],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def _run(fn):
    return asyncio.run(asyncio.wait_for(_session(fn), timeout=TIMEOUT))


def test_handshake_and_tool_list_over_stdio():
    """The protocol-level smoke test the in-process tests structurally cannot do.

    If ANY import in the server's path writes to stdout, initialize() raises a
    pydantic ValidationError here rather than in a user's client.
    """
    async def go(s):
        tools = await s.list_tools()
        return sorted(t.name for t in tools.tools)

    assert _run(go) == ["index_status", "list_papers", "search_papers"]


def test_search_round_trips_json_over_stdio():
    async def go(s):
        res = await s.call_tool("search_papers",
                                {"query": "POPE object hallucination", "k": 3})
        return json.loads(res.content[0].text)

    d = _run(go)
    assert d["count"] == 3
    assert d["mode"] == "hybrid"
    # Provenance survives serialisation — this is the thing that makes fusion
    # inspectable rather than asserted.
    assert any(r["dense_rank"] or r["bm25_rank"] for r in d["results"])
    assert all("UNTRUSTED PAPER EXCERPT" in r["excerpt"] for r in d["results"])


def test_injection_string_still_flagged_over_stdio():
    """The security claim, verified through the transport a client actually uses.

    'What is the capital of France?' is the literal injection string found in an
    indexed paper's appendix. The server cannot refuse on the caller's behalf,
    but it must report the gate signal truthfully.
    """
    async def go(s):
        res = await s.call_tool("search_papers",
                                {"query": "what is the capital of France?", "k": 2})
        return json.loads(res.content[0].text)

    d = _run(go)
    assert d["below_relevance_gate"] is True
    assert d["relevance"] < d["relevance_gate"]


def test_tools_declare_schemas():
    """A tool a client cannot introspect is a tool a model will call wrongly."""
    async def go(s):
        # SDK 2.0 renamed this to snake_case; `inputSchema` raises AttributeError.
        return {t.name: t.input_schema for t in (await s.list_tools()).tools}

    schemas = _run(go)
    props = schemas["search_papers"]["properties"]
    assert "query" in props
    assert set(props) >= {"query", "k", "mode"}
