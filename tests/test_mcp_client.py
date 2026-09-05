"""The MCP stdio client, against the fake server in fake_mcp_server.py."""
import threading
import time

import pytest

from conftest import load_plugin

reverie = load_plugin()
from reverie.mcp_client import (  # noqa: E402
    MCPClient, MCPError, MCPToolError, redact_env, server_env_from_environ,
)


def test_handshake_and_tools_list(client_factory):
    client = client_factory()
    client.start()
    assert client.is_running()
    assert client.server_info["name"] == "fake-reverie"
    names = [t["name"] for t in client.list_tools()]
    assert "search_memories" in names and "dream" in names


def test_call_tool_round_trip_parses_json(client_factory):
    client = client_factory()
    created = client.call_tool("create_memory", {"label": "Person", "properties": {"name": "Ada Lovelace"}})
    assert created["memory"]["name"] == "Ada Lovelace"
    assert created["memory"]["_labels"] == ["Person"]
    rows = client.call_tool("search_memories", {"query": "Ada", "limit": 5})
    assert [row["memory"]["name"] for row in rows] == ["Ada Lovelace"]


def test_start_is_lazy_and_idempotent(client_factory):
    client = client_factory()
    assert not client.is_running()
    client.call_tool("memory_stats", {})  # first call starts the server
    assert client.is_running()
    proc = client._proc
    client.start()
    assert client._proc is proc


def test_tool_error_is_surfaced(client_factory):
    client = client_factory(env={"FAKE_MCP_TOOL_ERROR": "dream"})
    with pytest.raises(MCPToolError, match="dream exploded"):
        client.call_tool("dream", {})
    assert client.call_tool("memory_stats", {})["nodes"] == 0  # the session survives a tool error


def test_unknown_tool_is_an_error(client_factory):
    client = client_factory()
    with pytest.raises(MCPToolError, match="Unknown tool"):
        client.call_tool("no_such_tool", {})


def test_crash_restarts_and_retries(client_factory):
    """The server dies mid-call: the client respawns it and the retry succeeds."""
    client = client_factory(env={"FAKE_MCP_CRASH_AFTER": "1"})
    client.call_tool("create_memory", {"label": "Person", "properties": {"name": "Grace Hopper"}})
    first_pid = client._proc.pid
    rows = client.call_tool("search_memories", {"query": "Grace"})  # crashes, restarts, retries
    assert [row["memory"]["name"] for row in rows] == ["Grace Hopper"]
    assert client.is_running() and client._proc.pid != first_pid


def test_dead_server_is_restarted_before_the_next_call(client_factory):
    client = client_factory()
    client.call_tool("memory_stats", {})
    client._proc.kill()
    client._proc.wait(timeout=5)
    assert client.call_tool("memory_stats", {})["nodes"] == 0
    assert client.is_running()


def test_timeout_when_the_server_never_answers(client_factory):
    client = client_factory(env={"FAKE_MCP_HANG": "1"}, timeout=0.5)
    with pytest.raises(MCPError, match="did not answer"):
        client.call_tool("memory_stats", {})


def test_startup_failure_reports_stderr(client_factory):
    client = client_factory(env={"FAKE_MCP_NO_START": "1"})
    with pytest.raises(MCPError) as excinfo:
        client.call_tool("memory_stats", {})
    assert "refusing to start" in str(excinfo.value) or "exited" in str(excinfo.value)


def test_missing_binary_is_reported():
    client = MCPClient(command="definitely-not-a-real-binary-xyz", timeout=2, startup_timeout=2)
    assert client.resolved_command() is None
    with pytest.raises(MCPError, match="not found"):
        client.start()


def test_ping_is_false_without_a_server():
    assert MCPClient(command="definitely-not-a-real-binary-xyz", startup_timeout=2).ping() is False


def test_env_passthrough_reaches_the_server(client_factory, monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("REVERIE_EMBEDDINGS", "none")
    monkeypatch.setenv("UNRELATED_VAR", "nope")
    env = server_env_from_environ({"REVERIE_MODEL_CACHE": "/tmp/models"})
    assert env["NEO4J_URI"] == "bolt://example:7687"
    assert env["REVERIE_EMBEDDINGS"] == "none"
    assert env["REVERIE_MODEL_CACHE"] == "/tmp/models"
    assert "UNRELATED_VAR" not in env

    env["FAKE_MCP_ECHO_ENV"] = "NEO4J_URI,REVERIE_EMBEDDINGS"
    client = client_factory(env=env)
    assert client.call_tool("echo_env", {}) == {"NEO4J_URI": "bolt://example:7687",
                                                "REVERIE_EMBEDDINGS": "none"}


def test_secrets_are_never_logged():
    safe = redact_env({"NEO4J_PASSWORD": "hunter2", "NEO4J_URI": "bolt://x", "OPENAI_API_KEY": "sk-1"})
    assert safe == {"NEO4J_PASSWORD": "***", "NEO4J_URI": "bolt://x", "OPENAI_API_KEY": "***"}


def test_concurrent_calls_are_serialised(client_factory):
    """Hermes may call from several turns at once; ids must not cross."""
    client = client_factory()
    client.start()
    results = {}
    errors = []

    def work(index):
        try:
            results[index] = client.call_tool(
                "create_memory", {"label": "Concept", "properties": {"name": f"Idea {index}"}})
        except Exception as exc:  # pragma: no cover - a failure here is the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors
    assert {r["memory"]["name"] for r in results.values()} == {f"Idea {i}" for i in range(8)}
    assert len({r["memory"]["_id"] for r in results.values()}) == 8


def test_empty_command_is_rejected():
    with pytest.raises(ValueError):
        MCPClient(command=[])


def test_a_timeout_with_a_live_server_is_never_retried(client_factory, call_log):
    """A slow server is still working: resending would apply the call twice.

    This is the cold-start case — the first search or create pays for the local embedding model —
    and it must hold for reads as well as writes, because the retry restarts the server and the
    model download starts over. The assertion is the server's own count of what reached it: a
    replay would show up as a second `create_memory`, however long the client waited.
    """
    for tool, args in (("create_memory", {"label": "Person", "properties": {"name": "Ava Walsh"}}),
                       ("search_memories", {"query": "Ava"})):
        # A fresh server per tool: the fake hangs inside the call, so a second request to the
        # same process would never be read and the count could not tell a replay from a silence.
        client = client_factory(env={"FAKE_MCP_HANG": "1", **call_log.env}, timeout=0.5)
        with pytest.raises(MCPError, match="still running"):
            client.call_tool(tool, args)
        assert call_log.count(tool) == 1, f"{tool} reached the server more than once"
        assert client.is_running(), "the server is left alone; its late answer is a stale id"
    assert call_log.tools() == ["create_memory", "search_memories"]


def test_a_timeout_is_reported_as_the_server_being_alive(client_factory):
    client = client_factory(env={"FAKE_MCP_HANG": "1"}, timeout=0.5)
    with pytest.raises(MCPError) as excinfo:
        client.call_tool("memory_stats", {})
    assert excinfo.value.server_gone is False and excinfo.value.delivered is True


def test_a_dead_server_is_restarted_and_a_read_retried(client_factory):
    """The process is gone, so a repeatable call can be sent again against a fresh server."""
    client = client_factory(env={"FAKE_MCP_CRASH_AFTER": "1"})
    client.call_tool("create_memory", {"label": "Person", "properties": {"name": "Grace Hopper"}})
    rows = client.call_tool("search_memories", {"query": "Grace"})  # dies, restarts, retries
    assert [row["memory"]["name"] for row in rows] == ["Grace Hopper"]


def test_a_write_is_not_retried_when_the_server_dies_mid_call(client_factory):
    """It may have committed before dying, so the caller is told rather than the write repeated."""
    client = client_factory(env={"FAKE_MCP_CRASH_AFTER": "1"})
    client.call_tool("memory_stats", {})
    with pytest.raises(MCPError, match="exited"):
        client.call_tool("create_memory", {"label": "Person", "properties": {"name": "Ada"}})


def test_a_write_is_retried_when_it_provably_never_left_the_client(client_factory):
    """The server was already dead when the call was made, so nothing could have been applied."""
    client = client_factory()
    client.call_tool("memory_stats", {})
    client._proc.kill()
    client._proc.wait(timeout=5)
    created = client.call_tool("create_memory", {"label": "Person", "properties": {"name": "Ada"}})
    assert created["memory"]["name"] == "Ada"


def test_read_only_tools_are_the_repeatable_ones():
    from reverie.mcp_client import READ_ONLY_TOOLS

    assert "search_memories" in READ_ONLY_TOOLS and "memory_stats" in READ_ONLY_TOOLS
    for write in ("create_memory", "update_memory", "create_connection", "delete_memory",
                  "delete_connection", "dream"):
        assert write not in READ_ONLY_TOOLS


def test_stale_responses_do_not_extend_the_timeout(client_factory):
    """A late answer to a timed-out call is skipped without buying the next call more time."""
    client = client_factory(env={"FAKE_MCP_HANG": "1"}, timeout=1.0)
    client.start()
    for stale_id in range(1, 6):  # answers to requests nobody is waiting for any more
        client._responses.put({"jsonrpc": "2.0", "id": -stale_id, "result": {}})
    started = time.monotonic()
    with pytest.raises(MCPError, match="did not answer"):
        client.call_tool("memory_stats", {})
    elapsed = time.monotonic() - started
    assert elapsed < 3, f"five stale messages must not buy five more timeouts (took {elapsed:.1f}s)"


def test_a_dead_server_replays_a_read_exactly_once(client_factory, call_log):
    """The dead-process path does retry — once, against the fresh server."""
    client = client_factory(env={"FAKE_MCP_CRASH_AFTER": "1", **call_log.env})
    client.call_tool("create_memory", {"label": "Person", "properties": {"name": "Ava Walsh"}})
    client.call_tool("search_memories", {"query": "Ava"})  # dies, restarts, retried once
    assert call_log.count("search_memories") == 2
    assert call_log.count("create_memory") == 1


def test_a_write_reaches_the_server_once_when_it_dies_mid_call(client_factory, call_log):
    client = client_factory(env={"FAKE_MCP_CRASH_AFTER": "1", **call_log.env})
    client.call_tool("memory_stats", {})
    with pytest.raises(MCPError, match="exited"):
        client.call_tool("create_memory", {"label": "Person", "properties": {"name": "Ava Walsh"}})
    assert call_log.count("create_memory") == 1, "a write is never replayed after a crash"
