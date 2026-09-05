"""Reverie tests. The provider imports Hermes internals (agent.memory_provider), which are
stubbed here so the plugin loads standalone; the live test needs the mcp-reverie server on PATH
and a Neo4j behind it (NEO4J_PASSWORD). The MCP client and the graph adapter are covered
without either, against tests/fake_mcp_server.py."""
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_plugin():
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


reverie = _load_plugin()
from reverie.graph import Graph, canonical_label  # noqa: E402


def test_terms_pick_names_emails_and_quotes():
    text = 'Ava Walsh from Atlantic Pharma emailed ava@atlanticpharma.example about the "support renewal" on Monday. Thanks!'
    terms = reverie.recall_terms(text)
    assert "ava@atlanticpharma.example" in terms
    assert "support renewal" in terms
    assert "Ava Walsh" in terms
    assert "Atlantic Pharma" in terms
    assert "Monday" not in terms and "Thanks" not in terms


def test_terms_dedupe_and_cap():
    names = ["Alice Smith", "Bob Jones", "Carol White", "Dan Brown", "Eve Black", "Frank Green",
             "Grace Hall", "Heidi King", "Ivan Lee", "Judy Moore"]
    text = ", ".join(names) + ", " + ", ".join(names)  # repeats must dedupe, then cap at 8
    terms = reverie.recall_terms(text)
    assert len(terms) == 8
    assert len(set(t.lower() for t in terms)) == 8
    assert reverie.recall_terms("") == []


def test_canonical_labels():
    assert canonical_label("person") == "Person"
    assert canonical_label("organisation") == "Organization"
    assert canonical_label("Company") == "Organization"
    assert canonical_label("Widget") == "Widget"
    with pytest.raises(ValueError):
        canonical_label("bad label!")


def test_provider_reports_unavailable_without_password(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    provider = reverie.ReverieMemoryProvider(config={})
    assert provider.name == "reverie"
    assert provider.is_available() is False
    assert provider.get_tool_schemas()[0]["name"] == "reverie"


@pytest.mark.skipif(
    not (os.environ.get("NEO4J_PASSWORD") and shutil.which("reverie")),
    reason="needs the mcp-reverie server on PATH and a live Neo4j behind it",
)
def test_live_roundtrip():
    g = Graph.from_env()
    assert g.ping()
    created = []
    try:
        node = g.remember("Person", "Reverie Test Person", {"role": "tester"})
        created.append(node["id"])
        assert node["props"]["name"] == "Reverie Test Person"
        org = g.remember("organisation", "Reverie Test Org")
        created.append(org["id"])
        assert org["label"] == "Organization"
        rel = g.connect("reverie test person", "REVERIE TEST ORG", "works_at", {"since": "2026"})
        assert rel["type"] == "WORKS_AT"
        hits = g.recall(["reverie test"], limit=5)
        assert any(h["props"]["name"] == "Reverie Test Person" for h in hits)
        assert any(r["type"] == "WORKS_AT" for h in hits for r in h["rels"])
        # remember again must update, not duplicate
        again = g.remember("person", "reverie test person", {"role": "lead tester"})
        assert again["id"] == node["id"]
        assert g.forget("Reverie Test Person") == 1
        assert not any(h["props"]["name"] == "Reverie Test Person" for h in g.recall(["reverie test person"]))
        report = g.dream(dry_run=True)
        assert {"relabelled", "merged", "orphans"} <= set(report)
    finally:
        # By node id: forget() resolves names through find(), which skips the node this test
        # archived, so a name-based cleanup would silently leave it behind.
        for node_id in created:
            g.call("delete_memory", nodeId=node_id)
        g.close()
