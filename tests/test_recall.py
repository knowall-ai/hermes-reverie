"""Automatic recall ranks by the server's relevance scores and shares its slots fairly across terms.

The regression this guards: on a voice call, "What do we know about Stephen Grimason?" recalled
nothing about Steven Grimason. The prompt's terms were ['What', 'Stephen Grimason', 'Ben Weeks'];
five weak semantic hits for "What" filled every recall slot.
"""
import json
import logging

import pytest

from conftest import load_plugin

reverie = load_plugin()
import reverie.graph as graph_module  # noqa: E402
from reverie.graph import DEFAULT_RECALL_MIN_SCORE, Graph, server_version_warning  # noqa: E402


def row(node_id, name, score, match, label="Person", matched=None, **props):
    memory = {**props, "name": name, "_id": node_id, "_labels": [label], "_score": score, "_match": match}
    if matched is not None:
        memory["_matched"] = matched
    return {"memory": memory, "connections": []}


class QueryClient:
    """Answers search_memories by query, like a graph would; records every call."""

    def __init__(self, answers, server_info=None):
        self.answers = answers
        self.calls = []
        self.server_info = server_info or {"name": "reverie", "version": "0.5.4"}

    def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return self.answers.get((arguments or {}).get("query"), [])

    def ping(self):
        return True

    def stop(self):
        pass


#: What the graph answered on the call, term by term (scores as the server reports them).
GRIMASON_GRAPH = {
    # A lone common word: weak semantic neighbours, nothing that is about "What".
    "What": [row(100 + i, f"Unrelated {i}", score, "semantic", label="Concept")
             for i, score in enumerate([0.42, 0.41, 0.4, 0.4, 0.4])],
    # The name, misheard: a fuzzy hit (0.5.4) on the stored spelling, plus a weak semantic one.
    "Stephen Grimason": [row(1, "Steven Grimason", 0.95, "fuzzy", matched="Steven Grimason", role="Director"),
                         row(2, "Ben Weeks", 0.32, "semantic")],
    "Ben Weeks": [row(2, "Ben Weeks", 1, "exact")],
}


def test_grimason_regression_the_name_is_recalled_and_the_noise_is_not():
    client = QueryClient(GRIMASON_GRAPH)
    hits = Graph(client).recall(["What", "Stephen Grimason", "Ben Weeks"], limit=5)

    names = [h["props"]["name"] for h in hits]
    assert names == ["Steven Grimason", "Ben Weeks"]
    assert hits[0]["match"] == "fuzzy" and hits[0]["matched"] == "Steven Grimason"
    # Names are searched before the single word.
    assert [c[1]["query"] for c in client.calls] == ["Stephen Grimason", "Ben Weeks", "What"]


def test_grimason_regression_through_prefetch():
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(QueryClient(GRIMASON_GRAPH))
    block = provider.prefetch("What do we know about Stephen Grimason? Ben Weeks asked.")
    assert "**Steven Grimason** (Director) [fuzzy name match, score 0.95]" in block
    assert "Unrelated" not in block


def test_a_noisy_term_that_scores_well_still_gets_only_its_share():
    """A common word can be a perfect keyword hit (it is a substring of some note); fair slots
    keep it from crowding out the names."""
    answers = {
        "What": [row(100 + i, f"Note {i}", 1, "keyword", label="Concept") for i in range(5)],
        "Stephen Grimason": [row(1, "Steven Grimason", 0.5, "keyword")],  # "Grimason" alone, an older server
        "Ben Weeks": [row(2, "Ben Weeks", 1, "exact")],
    }
    hits = Graph(QueryClient(answers)).recall(["What", "Stephen Grimason", "Ben Weeks"], limit=5)
    names = [h["props"]["name"] for h in hits]
    assert names[:2] == ["Steven Grimason", "Ben Weeks"]
    assert len(names) == 5 and names[2:] == ["Note 0", "Note 1", "Note 2"]


def test_slots_are_dealt_round_robin_and_deduplicated():
    answers = {
        "Ava Walsh": [row(1, "Ava Walsh", 1, "exact"), row(3, "Atlantic Pharma", 0.6, "semantic", "Organization")],
        "Atlantic Pharma": [row(3, "Atlantic Pharma", 1, "exact", "Organization"), row(4, "Atlantic Labs", 0.5, "keyword")],
    }
    hits = Graph(QueryClient(answers)).recall(["Ava Walsh", "Atlantic Pharma"], limit=3)
    assert [h["id"] for h in hits] == [1, 3, 4]  # Atlantic Pharma once, though both terms found it


def test_no_more_terms_are_searched_than_there_are_slots():
    client = QueryClient({})
    Graph(client).recall(["One", "Two Words", "Three", "Four More", "Five", "Six"], limit=3)
    assert [c[1]["query"] for c in client.calls] == ["Two Words", "Four More", "One"]


def test_explicit_search_is_not_filtered_and_carries_the_scores():
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(QueryClient(GRIMASON_GRAPH))
    results = json.loads(provider.handle_tool_call("reverie", {"action": "search", "query": "What"}))["results"]
    assert len(results) == 5 and {r["match"] for r in results} == {"semantic"}
    assert all(r["score"] < DEFAULT_RECALL_MIN_SCORE for r in results[2:])
    fuzzy = json.loads(provider.handle_tool_call(
        "reverie", {"action": "search", "query": "Stephen Grimason", "search_mode": "fuzzy"}))["results"][0]
    assert (fuzzy["score"], fuzzy["match"], fuzzy["matched"]) == (0.95, "fuzzy", "Steven Grimason")


def test_min_score_comes_from_config_and_is_validated():
    assert Graph(QueryClient({})).min_score == DEFAULT_RECALL_MIN_SCORE == 0.5
    graph = Graph.from_env({"recall_min_score": "0.7"})
    assert graph.min_score == 0.7
    assert Graph.from_env({"recall_min_score": ""}).min_score == DEFAULT_RECALL_MIN_SCORE
    for bad in ("high", 1.5, -0.1, float("nan")):
        with pytest.raises(ValueError, match="recall_min_score"):
            Graph.from_env({"recall_min_score": bad})
    answers = {"Ben Weeks": [row(2, "Ben Weeks", 1, "exact"), row(5, "Biscuit", 0.6, "semantic", "Pet")]}
    assert [h["id"] for h in Graph(QueryClient(answers), min_score=0.7).recall(["Ben Weeks"])] == [2]


def test_a_server_without_scores_keeps_every_hit_and_warns_once(caplog):
    unscored = {"memory": {"_id": 7, "_labels": ["Person"], "name": "Old Server Person"}, "connections": []}
    graph = Graph(QueryClient({"Old Server": [unscored], "Person": [unscored]}))
    with caplog.at_level(logging.WARNING, logger=graph_module.__name__):
        assert [h["id"] for h in graph.recall(["Old Server"])] == [7]
        assert [h["id"] for h in graph.recall(["Person"])] == [7]
    warnings = [r for r in caplog.records if "without a score" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("version, warns", [
    ("0.5.4", False), ("0.6.0", False), ("1.0.0", False), ("0.5.4-beta.1", False),
    ("0.5.3", True), ("0.5.0", True), ("0.4.0", True), ("unknown", True), (None, True),
])
def test_server_version_warning(version, warns):
    warning = server_version_warning({"name": "reverie", "version": version})
    assert bool(warning) is warns
    if warns:
        assert "0.5.4" in warning


def test_initialize_warns_about_an_old_server_but_still_connects(reverie, fake_command, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_MCP_SERVER_VERSION", "0.5.0")  # what every build before 0.5.4 reported
    provider = reverie.ReverieMemoryProvider(config={"server_command": " ".join(fake_command)})
    with caplog.at_level(logging.WARNING):
        provider.initialize("session")
    try:
        assert provider._graph is not None
        assert any("mcp-reverie 0.5.0 is older than 0.5.4" in r.getMessage() for r in caplog.records)
    finally:
        provider.shutdown()


def test_initialize_is_quiet_with_a_current_server(reverie, fake_command, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_MCP_SERVER_VERSION", "0.5.4")
    provider = reverie.ReverieMemoryProvider(config={"server_command": " ".join(fake_command)})
    with caplog.at_level(logging.WARNING):
        provider.initialize("session")
    try:
        assert not [r for r in caplog.records if "older than" in r.getMessage()]
    finally:
        provider.shutdown()


def test_the_agent_is_told_to_search_for_names_recall_missed():
    description = reverie.REVERIE_TOOL["description"]
    assert "fuzzy" in description and "surname" in description and "matched" in description
    assert "fuzzy" in reverie.REVERIE_TOOL["parameters"]["properties"]["search_mode"]["enum"]
