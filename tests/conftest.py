"""Shared fixtures: load the plugin standalone and point it at the fake MCP server."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
FAKE_SERVER = Path(__file__).resolve().parent / "fake_mcp_server.py"


def load_plugin():
    """Import the plugin as the ``reverie`` package, stubbing the Hermes internals it imports."""
    if "reverie" in sys.modules:
        return sys.modules["reverie"]
    agent = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # minimal stand-in
        pass

    class RecallStatus:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    mp.MemoryProvider = MemoryProvider
    mp.RecallStatus = RecallStatus
    mp.is_trivial_prompt = lambda t: not (t or "").strip()
    agent.memory_provider = mp
    sys.modules.setdefault("agent", agent)
    sys.modules.setdefault("agent.memory_provider", mp)
    spec = importlib.util.spec_from_file_location(
        "reverie", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["reverie"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def reverie():
    return load_plugin()


@pytest.fixture
def fake_command():
    """The command that starts the fake MCP server."""
    return [sys.executable, str(FAKE_SERVER)]


@pytest.fixture
def client_factory(reverie, fake_command, tmp_path):
    """Build MCPClients against the fake server, all sharing one persisted graph."""
    from reverie.mcp_client import MCPClient

    created = []

    def make(env=None, **kwargs):
        server_env = {"FAKE_MCP_STATE": str(tmp_path / "graph.json")}
        server_env.update(env or {})
        client = MCPClient(command=fake_command, env=server_env, timeout=kwargs.pop("timeout", 10),
                           startup_timeout=kwargs.pop("startup_timeout", 10), **kwargs)
        created.append(client)
        return client

    yield make
    for client in created:
        client.stop()


@pytest.fixture
def call_log(tmp_path):
    """Every tools/call the fake server received, in order — including ones it never answered."""
    path = tmp_path / "calls.log"

    class CallLog:
        env = {"FAKE_MCP_CALL_LOG": str(path)}

        @staticmethod
        def tools():
            return path.read_text(encoding="utf-8").split() if path.exists() else []

        @staticmethod
        def count(tool=None):
            names = CallLog.tools()
            return len([n for n in names if tool is None or n == tool])

    return CallLog


@pytest.fixture
def graph(reverie, client_factory):
    from reverie.graph import Graph

    return Graph(client_factory())
