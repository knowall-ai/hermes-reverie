"""graph.py as a thin adapter: every verb maps onto the MCP tools and comes back in the shape
the provider renders. Run against the fake MCP server, so no Neo4j is needed."""
import json

import pytest

from conftest import load_plugin

reverie = load_plugin()
import reverie.graph as graph_module  # noqa: E402
from reverie.graph import Graph, canonical_label  # noqa: E402
from reverie.mcp_client import MCPToolError  # noqa: E402


class RecordingClient:
    """A client that records calls and replays canned results — for mapping assertions."""

    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return self.results.pop(0) if self.results else {}

    def stop(self):
        pass

    def ping(self):
        return True


def memory(node_id, label, **props):
    return {**props, "_id": node_id, "_labels": [label]}


# -- mapping -----------------------------------------------------------------
def test_recall_maps_to_search_memories_with_one_call():
    client = RecordingClient([[{"memory": memory(1, "Person", name="Ava Walsh", role="CEO", _score=1.0,
                                                 _match="keyword"),
                                "connections": [{"memory": memory(2, "Organization", name="Atlantic Pharma"),
                                                 "relationship": {"_id": 9, "_type": "WORKS_AT"},
                                                 "distance": 1}]}]])
    graph = Graph(client, search_mode="hybrid", similarity_threshold=0.55, depth=2)
    hits = graph.recall(["Ava Walsh", "ava@atlanticpharma.example"], limit=5)

    assert len(client.calls) == 1, "recall must be a single round trip"
    tool, args = client.calls[0]
    assert tool == "search_memories"
    assert args["query"] == "Ava Walsh ava@atlanticpharma.example"
    assert args["search_mode"] == "hybrid" and args["similarity_threshold"] == 0.55
    assert args["depth"] == 2 and args["limit"] == 10  # asks for 2x the limit, then filters

    assert hits[0]["label"] == "Person"
    assert hits[0]["props"] == {"name": "Ava Walsh", "role": "CEO"}  # _-prefixed metadata stripped
    assert hits[0]["score"] == 1.0 and hits[0]["match"] == "keyword"
    assert hits[0]["rels"] == [{"type": "WORKS_AT", "out": True, "name": "Atlantic Pharma",
                                "label": "Organization", "distance": 1}]


def test_recall_without_terms_calls_nothing():
    client = RecordingClient()
    assert Graph(client).recall(["", "  "]) == []
    assert client.calls == []


def test_recall_drops_archived_and_honours_the_limit():
    rows = [{"memory": memory(i, "Person", name=f"P{i}", **({"status": "archived"} if i % 2 else {})),
             "connections": []} for i in range(1, 7)]
    graph = Graph(RecordingClient([rows]))
    hits = graph.recall(["p"], limit=2)
    assert [h["props"]["name"] for h in hits] == ["P2", "P4"]


def test_probe_is_a_recall_of_three():
    client = RecordingClient([[]])
    Graph(client).probe(" Atlantic ")
    assert client.calls[0][1]["query"] == "Atlantic" and client.calls[0][1]["limit"] == 6


def test_remember_creates_when_nothing_matches():
    client = RecordingClient([[], {"memory": memory(7, "Person", name="Ada Lovelace", role="analyst")}])
    node = Graph(client).remember("person", " Ada Lovelace ", {"role": "analyst", "name": "ignored", "x": None})

    lookup, create = client.calls
    assert lookup[0] == "search_memories"
    assert lookup[1]["search_mode"] == "keyword" and lookup[1]["depth"] == 0
    assert create == ("create_memory", {"label": "Person", "properties": {"name": "Ada Lovelace", "role": "analyst"}})
    assert node == {"id": 7, "label": "Person", "props": {"name": "Ada Lovelace", "role": "analyst"}}


def test_remember_updates_the_existing_node_instead_of_duplicating():
    existing = [{"memory": memory(3, "Person", name="ada lovelace"), "connections": []}]
    client = RecordingClient([existing, {"memory": memory(3, "Person", name="ada lovelace", role="lead")}])
    node = Graph(client).remember("Person", "Ada Lovelace", {"role": "lead"})

    assert client.calls[1][0] == "update_memory"
    assert client.calls[1][1]["nodeId"] == 3
    assert client.calls[1][1]["properties"]["role"] == "lead"
    assert "updated_at" in client.calls[1][1]["properties"]
    assert node["id"] == 3


def test_remember_passes_on_the_servers_bloat_hint():
    client = RecordingClient([[{"memory": memory(3, "Person", name="Ben"), "connections": []}],
                              {"memory": {**memory(3, "Person", name="Ben"), "_hint": "too many properties"}}])
    assert Graph(client).remember("Person", "Ben", {"a": 1})["hint"] == "too many properties"


def test_remember_needs_a_name():
    with pytest.raises(ValueError):
        Graph(RecordingClient()).remember("Person", "   ")


def test_connect_resolves_both_names_then_creates_the_connection():
    client = RecordingClient([
        [{"memory": memory(1, "Person", name="Ava Walsh"), "connections": []}],
        [{"memory": memory(2, "Organization", name="Atlantic Pharma"), "connections": []}],
        {"relationship": {"_id": 9, "_type": "WORKS_AT"}},
    ])
    result = Graph(client).connect("ava walsh", "ATLANTIC PHARMA", "works_at", {"since": "2026"})

    assert client.calls[2][0] == "create_connection"
    assert client.calls[2][1]["fromMemoryId"] == 1 and client.calls[2][1]["toMemoryId"] == 2
    assert client.calls[2][1]["type"] == "WORKS_AT"
    assert client.calls[2][1]["properties"]["since"] == "2026"
    assert result == {"from": "Ava Walsh", "type": "WORKS_AT", "to": "Atlantic Pharma", "props": {"since": "2026"}}


def test_connect_refuses_unknown_entities():
    client = RecordingClient([[], []])
    with pytest.raises(RuntimeError, match="remember it first"):
        Graph(client).connect("Nobody", "Nowhere", "KNOWS")


def test_connect_validates_the_relationship_type():
    with pytest.raises(ValueError):
        Graph(RecordingClient()).connect("a", "b", "not a type!")


def test_forget_archives_by_default_and_deletes_when_asked():
    found = [{"memory": memory(4, "Person", name="Old Contact"), "connections": []}]
    soft = RecordingClient([found, {"memory": memory(4, "Person", name="Old Contact", status="archived")}])
    assert Graph(soft).forget("Old Contact") == 1
    assert soft.calls[1][0] == "update_memory"
    assert soft.calls[1][1]["properties"]["status"] == "archived"

    hard = RecordingClient([found, {"deletedCount": 1}])
    assert Graph(hard).forget("Old Contact", hard=True) == 1
    assert hard.calls[1][0] == "delete_memory" and hard.calls[1][1] == {"nodeId": 4}


def test_forget_unknown_name_is_zero():
    client = RecordingClient([[]])
    assert Graph(client).forget("Ghost") == 0
    assert len(client.calls) == 1


def test_forget_connection_maps_to_delete_connection():
    client = RecordingClient([
        [{"memory": memory(1, "Person", name="A"), "connections": []}],
        [{"memory": memory(2, "Person", name="B"), "connections": []}],
        {"deletedCount": 1},
    ])
    assert Graph(client).forget_connection("A", "B", "knows") == 1
    assert client.calls[2] == ("delete_connection", {"fromMemoryId": 1, "toMemoryId": 2, "type": "KNOWS"})


def test_cypher_maps_to_query_memories_and_refuses_writes():
    client = RecordingClient([[{"n": 1}]])
    graph = Graph(client)
    assert graph.read_cypher("MATCH (n) RETURN n", {"x": 1}) == [{"n": 1}]
    assert client.calls[0] == ("query_memories", {"cypher": "MATCH (n) RETURN n", "params": {"x": 1}})
    with pytest.raises(ValueError, match="read-only"):
        graph.read_cypher("MATCH (n) DETACH DELETE n")


def test_stats_and_dream_pass_straight_through():
    client = RecordingClient([{"nodes": 12, "labels": {"Person": 12}}, {"relabelled": 0, "merged": 1}])
    graph = Graph(client)
    assert graph.stats()["nodes"] == 12
    assert client.calls[0] == ("memory_stats", {})
    assert graph.dream(dry_run=True)["merged"] == 1
    assert client.calls[1] == ("dream", {"dry_run": True})


def test_search_arguments_are_clamped():
    client = RecordingClient([[]])
    Graph(client).search_memories("x", limit=9999, depth=99)
    assert client.calls[0][1]["limit"] == 200 and client.calls[0][1]["depth"] == 5


def test_from_env_reads_config(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "hunter2")
    graph = Graph.from_env({"server_command": "reverie --flag", "embeddings": "none",
                            "search_mode": "semantic", "similarity_threshold": 0.7, "recall_depth": 3})
    assert graph.client.command == ["reverie", "--flag"]
    assert graph.client._extra_env["REVERIE_EMBEDDINGS"] == "none"
    assert graph.client._extra_env["NEO4J_PASSWORD"] == "hunter2"
    assert (graph.search_mode, graph.similarity_threshold, graph.depth) == ("semantic", 0.7, 3)


# -- end to end against the fake server --------------------------------------
def test_round_trip_against_the_fake_server(graph):
    person = graph.remember("person", "Reverie Test Person", {"role": "tester"})
    org = graph.remember("organisation", "Reverie Test Org")
    assert org["label"] == "Organization"

    graph.connect("reverie test person", "REVERIE TEST ORG", "works_at", {"since": "2026"})
    hits = graph.recall(["reverie test"], limit=5)
    assert any(h["props"]["name"] == "Reverie Test Person" for h in hits)
    assert any(r["type"] == "WORKS_AT" for h in hits for r in h["rels"])

    # remembering again updates, it does not duplicate
    again = graph.remember("person", "reverie test person", {"role": "lead tester"})
    assert again["id"] == person["id"]
    assert again["props"]["role"] == "lead tester"

    assert graph.stats()["nodes"] == 2
    assert graph.forget("Reverie Test Person") == 1
    assert not any(h["props"]["name"] == "Reverie Test Person" for h in graph.recall(["reverie test person"]))
    assert set(graph.dream(dry_run=True)) >= {"relabelled", "merged", "orphans", "duplicates"}
    graph.close()


def test_tool_errors_surface_from_the_fake_server(reverie, client_factory):
    graph = Graph(client_factory(env={"FAKE_MCP_TOOL_ERROR": "memory_stats"}))
    with pytest.raises(MCPToolError):
        graph.stats()


def test_provider_tool_call_reports_errors_as_json(reverie, client_factory):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(client_factory(env={"FAKE_MCP_TOOL_ERROR": "memory_stats"}))
    assert "error" in json.loads(provider.handle_tool_call("reverie", {"action": "stats"}))


def test_provider_actions_against_the_fake_server(reverie, client_factory):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(client_factory())

    remembered = json.loads(provider.handle_tool_call(
        "reverie", {"action": "remember", "label": "Person", "name": "Ada Lovelace", "properties": {"role": "analyst"}}))
    assert remembered["remembered"]["props"]["name"] == "Ada Lovelace"

    json.loads(provider.handle_tool_call(
        "reverie", {"action": "remember", "label": "Organization", "name": "Analytical Engines"}))
    connected = json.loads(provider.handle_tool_call(
        "reverie", {"action": "connect", "from": "Ada Lovelace", "to": "Analytical Engines", "type": "WORKS_AT"}))
    assert connected["connected"]["type"] == "WORKS_AT"

    found = json.loads(provider.handle_tool_call(
        "reverie", {"action": "search", "query": "Ada", "search_mode": "keyword", "limit": 5}))
    assert found["results"][0]["props"]["name"] == "Ada Lovelace"

    assert json.loads(provider.handle_tool_call("reverie", {"action": "stats"}))["counts"]["nodes"] == 2
    assert json.loads(provider.handle_tool_call("reverie", {"action": "dream", "dry_run": True}))["dream"]["dry_run"]
    assert json.loads(provider.handle_tool_call(
        "reverie", {"action": "forget", "from": "Ada Lovelace", "to": "Analytical Engines",
                    "type": "WORKS_AT"}))["disconnected"] == 1
    # archiving last: an archived entity is no longer resolvable by name, by design
    assert json.loads(provider.handle_tool_call("reverie", {"action": "forget", "name": "Ada Lovelace"}))["archived"] == 1
    assert "error" in json.loads(provider.handle_tool_call("reverie", {"action": "nonsense"}))


def test_prefetch_renders_recalled_entities(reverie, client_factory):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(client_factory())
    provider._graph.remember("Person", "Ava Walsh", {"role": "CEO", "company": "Atlantic Pharma"})

    block = provider.prefetch("Any news from Ava Walsh?")
    assert block.startswith("## Reverie recalls")
    assert "Person **Ava Walsh** (CEO, Atlantic Pharma)" in block
    assert provider.recall_status().count == 1
    assert provider.prefetch("") == ""


def test_labels_still_canonicalise():
    assert canonical_label("company") == "Organization"


# -- the server's argument contract (mcp-reverie @ 01dceae) -------------------
@pytest.mark.parametrize("raw, expected", [
    ("person", "Person"), ("PERSON", "Person"), ("organisation", "Organization"),
    ("company", "Organization"), ("companies", "Organization"), ("org", "Organization"),
    ("Widget", "Widget"), ("widget", "Widget"), ("", "Concept"),
])
def test_labels_are_canonicalised_to_identifiers(raw, expected):
    label = canonical_label(raw)
    assert label == expected
    assert graph_module.IDENT_RE.match(label), "a label is interpolated into Cypher; it must be an identifier"


@pytest.mark.parametrize("bad", ["Atlantic Pharma", "bad label!", "9Lives", "a" * 65, "Person; DROP"])
def test_free_text_labels_are_refused(bad):
    with pytest.raises(ValueError, match="invalid label"):
        canonical_label(bad)


def test_remember_refuses_a_free_text_label():
    client = RecordingClient([[]])
    with pytest.raises(ValueError, match="invalid label"):
        Graph(client).remember("Atlantic Pharma", "Ava Walsh")
    assert client.calls == [], "nothing reaches the server once the label is rejected"


@pytest.mark.parametrize("props", [
    {"aliases": {"nested": "object"}},
    {"history": [{"a": 1}]},
    {"tags": {"a", "b"}},
])
def test_remember_refuses_property_values_the_graph_cannot_store(props):
    client = RecordingClient([[]])
    with pytest.raises(ValueError, match="model anything richer"):
        Graph(client).remember("Person", "Ada Lovelace", props)


def test_remember_accepts_primitives_and_lists_of_primitives():
    client = RecordingClient([[], {"memory": memory(1, "Person", name="Ada Lovelace")}])
    Graph(client).remember("Person", "Ada Lovelace", {"age": 36, "vip": True, "aliases": ["Ada", "A.L."]})
    assert client.calls[1][1]["properties"]["aliases"] == ["Ada", "A.L."]


def test_remember_refuses_non_object_properties():
    with pytest.raises(ValueError, match="must be an object"):
        Graph(RecordingClient([[]])).remember("Person", "Ada", ["role", "analyst"])


@pytest.mark.parametrize("bad_id", [-1, 1.5, "3", True, None])
def test_node_ids_must_be_non_negative_integers(bad_id):
    with pytest.raises(ValueError, match="non-negative integer"):
        graph_module._node_id(bad_id)


def test_forget_refuses_a_bad_node_id_from_the_server():
    client = RecordingClient([[{"memory": {"name": "Ada", "_id": -1, "_labels": ["Person"]}, "connections": []}]])
    with pytest.raises(ValueError, match="non-negative integer"):
        Graph(client).forget("Ada")


@pytest.mark.parametrize("bad", ["not a type!", "9LIVES", "", "WORKS AT"])
def test_relationship_types_must_be_identifiers(bad):
    with pytest.raises(ValueError, match="invalid relationship type"):
        Graph(RecordingClient()).connect("a", "b", bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("inf"), float("nan"), "close"])
def test_similarity_threshold_must_be_finite_and_in_range(bad):
    with pytest.raises(ValueError, match="similarity_threshold"):
        Graph(RecordingClient([[]])).search_memories("x", similarity_threshold=bad)


def test_search_mode_is_validated():
    with pytest.raises(ValueError, match="invalid search_mode"):
        Graph(RecordingClient([[]])).search_memories("x", search_mode="fuzzy")


def test_search_sends_only_keys_the_server_allows():
    client = RecordingClient([[]])
    Graph(client).search_memories("x", limit=5)
    allowed = {"query", "label", "depth", "order_by", "limit", "since_date", "search_mode",
               "similarity_threshold"}
    assert set(client.calls[0][1]) <= allowed


@pytest.mark.parametrize("cypher, reason", [
    ("MATCH (n) DETACH DELETE n", "DETACH"),
    ("MATCH (n) /* sneaky */ SET n.x = 1", "SET"),
    ("MATCH (n) // comment\nCREATE (m)", "CREATE"),
    ("CALL { MATCH (n) RETURN n } RETURN 1", "subqueries"),
    ("CALL apoc.refactor.mergeNodes([]) YIELD node RETURN node", "allow-list"),
    ("MATCH (n) FOREACH (x IN [] | SET n.y = 1)", "FOREACH"),
])
def test_cypher_guard_matches_the_servers(cypher, reason):
    with pytest.raises(ValueError, match=reason):
        Graph(RecordingClient()).read_cypher(cypher)


@pytest.mark.parametrize("cypher", [
    "MATCH (n:Person) RETURN n.name LIMIT 10",
    "CALL db.labels() YIELD label RETURN label",
    "// just a comment\nMATCH (n) RETURN count(n)",
])
def test_read_only_cypher_is_allowed(cypher):
    client = RecordingClient([[]])
    Graph(client).read_cypher(cypher)
    assert client.calls[0][0] == "query_memories"


def test_empty_cypher_is_refused():
    with pytest.raises(ValueError, match="needs a query"):
        Graph(RecordingClient()).read_cypher("   ")


def test_from_env_refuses_an_unknown_embeddings_provider():
    with pytest.raises(ValueError, match="invalid embeddings provider"):
        Graph.from_env({"embeddings": "magic"})


@pytest.mark.parametrize("provider", ["local", "openai", "azure", "ollama", "voyage", "none"])
def test_from_env_accepts_every_provider_the_server_supports(provider):
    graph = Graph.from_env({"embeddings": provider})
    assert graph.client._extra_env["REVERIE_EMBEDDINGS"] == provider


def test_the_fake_server_rejects_unknown_keys(client_factory):
    """The fake enforces the same argument contract, so a drifting client fails here first."""
    client = client_factory()
    with pytest.raises(MCPToolError, match="unknown key"):
        client.call_tool("memory_stats", {"nope": 1})
    with pytest.raises(MCPToolError, match="Invalid Cypher identifier"):
        client.call_tool("create_memory", {"label": "Atlantic Pharma", "properties": {"name": "x"}})
    with pytest.raises(MCPToolError, match="Invalid nodeId"):
        client.call_tool("delete_memory", {"nodeId": -1})


def test_from_env_rejects_a_bad_search_mode_once_rather_than_on_every_recall():
    with pytest.raises(ValueError, match="invalid search_mode"):
        Graph.from_env({"search_mode": "fuzzy"})


def test_from_env_rejects_an_out_of_range_similarity_threshold():
    with pytest.raises(ValueError, match="similarity_threshold"):
        Graph.from_env({"similarity_threshold": 2.5})


def test_from_env_accepts_a_blank_similarity_threshold():
    assert Graph.from_env({"similarity_threshold": ""}).similarity_threshold is None


def test_writes_are_not_repeatable_but_searches_are(client_factory):
    """graph.py goes through call_tool, so the retry policy follows the tool, not the caller."""
    calls = []
    graph = Graph(client_factory())
    original = graph.client.call_tool

    def spy(name, arguments=None, **kwargs):
        calls.append((name, kwargs.get("repeatable")))
        return original(name, arguments, **kwargs)

    graph.client.call_tool = spy
    graph.remember("Person", "Ada Lovelace")
    assert [name for name, _ in calls] == ["search_memories", "create_memory"]
    assert all(flag is None for _, flag in calls), "the client decides from the tool name"


def test_unparseable_server_command_is_reported(reverie):
    provider = reverie.ReverieMemoryProvider(config={"server_command": 'reverie "'})
    assert provider.is_available() is False
    assert "quoting" in provider.unavailable_reason()


def test_config_schema_covers_every_persisted_key(reverie):
    provider = reverie.ReverieMemoryProvider(config={})
    keys = {entry["key"] for entry in provider.get_config_schema()}
    assert {"similarity_threshold", "recall_depth", "search_mode", "server_command",
            "embeddings"} <= keys


# -- exact-name resolution and ambiguity -------------------------------------
def named(node_id, label, name):
    return {"memory": memory(node_id, label, name=name), "connections": []}


def test_name_lookup_asks_for_the_whole_page_so_a_common_name_cannot_crowd_it_out():
    """A keyword search for "James" can match hundreds of nodes; the real one must be in the page."""
    client = RecordingClient([[]])
    Graph(client).find("James Kelly", label="Person")
    tool, args = client.calls[0]
    assert tool == "search_memories"
    assert args["limit"] == 200, "the server's maximum, until search_mode 'exact' exists"
    assert args["depth"] == 0 and args["search_mode"] == "keyword"
    assert args["label"] == "Person", "the label is pushed down so the page holds only candidates"


def test_the_name_lookup_has_one_switch_point_for_the_future_exact_mode():
    assert set(graph_module.NAME_SEARCH) == {"search_mode", "limit", "depth"}
    assert graph_module.NAME_SEARCH["search_mode"] in graph_module.SEARCH_MODES


def test_find_all_returns_every_exact_match_and_ignores_near_ones():
    rows = [named(1, "Person", "James Kelly"), named(2, "Person", "James Kelly Jr"),
            named(3, "Organization", "james kelly"), named(4, "Person", "James Kelly")]
    matches = Graph(RecordingClient([rows])).find_all("james kelly")
    assert [m["id"] for m in matches] == [1, 3, 4]


def test_find_all_filters_by_label():
    rows = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    assert [m["id"] for m in Graph(RecordingClient([rows])).find_all("Atlas", label="Project")] == [2]


def test_find_all_skips_archived():
    rows = [{"memory": memory(1, "Person", name="Ada", status="archived"), "connections": []},
            named(2, "Person", "Ada")]
    assert [m["id"] for m in Graph(RecordingClient([rows])).find_all("Ada")] == [2]


def test_resolve_refuses_an_ambiguous_name():
    rows = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    with pytest.raises(RuntimeError, match="ambiguous"):
        Graph(RecordingClient([rows])).resolve("Atlas")


def test_resolve_names_the_labels_it_could_not_choose_between():
    rows = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    with pytest.raises(RuntimeError) as excinfo:
        Graph(RecordingClient([rows])).resolve("Atlas")
    assert "Person, Project" in str(excinfo.value) and "passing a label" in str(excinfo.value)


def test_resolve_accepts_a_label_to_break_the_tie():
    rows = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    assert Graph(RecordingClient([rows, rows])).resolve("Atlas", "Project")["id"] == 2


def test_connect_refuses_an_ambiguous_end_rather_than_guessing():
    """Writing WORKS_AT to the wrong Atlas is worse than not writing it."""
    ambiguous = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    client = RecordingClient([ambiguous])
    with pytest.raises(RuntimeError, match="connect: source 'Atlas' is ambiguous"):
        Graph(client).connect("Atlas", "KnowAll", "WORKS_AT")
    assert [name for name, _ in client.calls] == ["search_memories"], "nothing was written"


def test_connect_takes_labels_for_both_ends():
    ambiguous = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    orgs = [named(3, "Person", "KnowAll"), named(4, "Organization", "KnowAll")]
    client = RecordingClient([ambiguous, orgs, {"relationship": {"_id": 9, "_type": "OWNS"}}])
    result = Graph(client).connect("Atlas", "KnowAll", "OWNS",
                                   from_label="Project", to_label="Organization")
    assert client.calls[0][1]["label"] == "Project" and client.calls[1][1]["label"] == "Organization"
    assert client.calls[2][1]["fromMemoryId"] == 2 and client.calls[2][1]["toMemoryId"] == 4
    assert result["type"] == "OWNS"


def test_forget_connection_refuses_an_ambiguous_end():
    ambiguous = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    with pytest.raises(RuntimeError, match="ambiguous"):
        Graph(RecordingClient([ambiguous])).forget_connection("Atlas", "KnowAll", "OWNS")


def test_forget_connection_is_a_no_op_when_an_end_is_unknown():
    client = RecordingClient([[]])
    assert Graph(client).forget_connection("Ghost", "KnowAll", "OWNS") == 0
    assert len(client.calls) == 1


def test_forget_connection_takes_labels():
    ambiguous = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    orgs = [named(4, "Organization", "KnowAll")]
    client = RecordingClient([ambiguous, orgs, {"deletedCount": 1}])
    assert Graph(client).forget_connection("Atlas", "KnowAll", "owns", from_label="Project") == 1
    assert client.calls[2] == ("delete_connection",
                               {"fromMemoryId": 2, "toMemoryId": 4, "type": "OWNS"})


def test_the_tool_passes_labels_through(reverie, client_factory):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(client_factory())
    for label in ("Person", "Project"):
        provider.handle_tool_call("reverie", {"action": "remember", "label": label, "name": "Atlas"})
    provider.handle_tool_call("reverie", {"action": "remember", "label": "Organization", "name": "KnowAll"})

    refused = json.loads(provider.handle_tool_call(
        "reverie", {"action": "connect", "from": "Atlas", "to": "KnowAll", "type": "OWNED_BY"}))
    assert "ambiguous" in refused["error"]

    connected = json.loads(provider.handle_tool_call(
        "reverie", {"action": "connect", "from": "Atlas", "to": "KnowAll", "type": "OWNED_BY",
                    "from_label": "Project"}))
    assert connected["connected"] == {"from": "Atlas", "type": "OWNED_BY", "to": "KnowAll", "props": {}}

    disconnected = json.loads(provider.handle_tool_call(
        "reverie", {"action": "forget", "from": "Atlas", "to": "KnowAll", "type": "OWNED_BY",
                    "from_label": "Project"}))
    assert disconnected["disconnected"] == 1


def test_the_tool_schema_advertises_the_label_arguments(reverie):
    schema = reverie.ReverieMemoryProvider(config={}).get_tool_schemas()[0]
    assert {"from_label", "to_label"} <= set(schema["parameters"]["properties"])


# -- resolve() outcomes are types, not message text ---------------------------
def test_resolve_raises_a_typed_not_found():
    with pytest.raises(graph_module.MemoryNotFound) as excinfo:
        Graph(RecordingClient([[]])).resolve("Ava Walsh", "Person")
    assert excinfo.value.name == "Ava Walsh" and excinfo.value.label == "Person"


def test_resolve_raises_a_typed_ambiguity_carrying_the_candidates():
    rows = [named(1, "Person", "Atlas"), named(2, "Project", "Atlas")]
    with pytest.raises(graph_module.AmbiguousMemory) as excinfo:
        Graph(RecordingClient([rows])).resolve("Atlas")
    assert [m["id"] for m in excinfo.value.matches] == [1, 2]


def test_an_entity_literally_called_not_found_is_still_refused_when_ambiguous():
    """The old substring check on the message would have swallowed this one."""
    rows = [named(1, "Person", "not found"), named(2, "Project", "not found")]
    with pytest.raises(graph_module.AmbiguousMemory):
        Graph(RecordingClient([rows])).forget_connection("not found", "Atlantic Pharma", "OWNS")


def test_an_entity_literally_called_not_found_still_disconnects_when_unique():
    client = RecordingClient([[named(1, "Person", "not found")],
                              [named(2, "Organization", "Atlantic Pharma")],
                              {"deletedCount": 1}])
    assert Graph(client).forget_connection("not found", "Atlantic Pharma", "OWNS") == 1


# -- the forget action must not mistake half a relationship for an entity -----
@pytest.mark.parametrize("partial", [
    {"from": "Ava Walsh"},
    {"to": "Atlantic Pharma"},
    {"type": "WORKS_AT"},
    {"from": "Ava Walsh", "to": "Atlantic Pharma"},
    {"from": "Ava Walsh", "type": "WORKS_AT"},
    {"to": "Atlantic Pharma", "type": "WORKS_AT"},
])
def test_forget_refuses_half_a_relationship_instead_of_archiving_an_entity(reverie, partial):
    """A partial edge must never fall through to the entity delete."""
    provider = reverie.ReverieMemoryProvider(config={})
    client = RecordingClient()
    provider._graph = Graph(client)
    result = json.loads(provider.handle_tool_call(
        "reverie", {"action": "forget", "name": "Ava Walsh", **partial}))
    assert "from" in result["error"] and "to" in result["error"] and "type" in result["error"]
    assert client.calls == [], "nothing was archived or deleted"


def test_forget_needs_a_name_when_no_relationship_is_given(reverie):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(RecordingClient())
    result = json.loads(provider.handle_tool_call("reverie", {"action": "forget"}))
    assert "needs 'name'" in result["error"]


def test_forget_still_archives_a_named_entity(reverie, client_factory):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(client_factory())
    provider.handle_tool_call("reverie", {"action": "remember", "label": "Person", "name": "Ava Walsh"})
    assert json.loads(provider.handle_tool_call(
        "reverie", {"action": "forget", "name": "Ava Walsh"}))["archived"] == 1


# -- boolean tool arguments are validated, never coerced ----------------------
@pytest.fixture
def provider_on_fake(reverie, client_factory):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(client_factory())
    return provider


def _remember(provider, name="Ava Walsh", label="Person"):
    return json.loads(provider.handle_tool_call(
        "reverie", {"action": "remember", "label": label, "name": name}))["remembered"]["id"]


@pytest.mark.parametrize("truthy_junk", ["false", "FALSE", " False "])
def test_hard_false_as_a_string_archives_and_does_not_delete(provider_on_fake, truthy_junk):
    """bool("false") is True: the old coercion turned a soft archive into a permanent delete."""
    node_id = _remember(provider_on_fake)
    result = json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "forget", "name": "Ava Walsh", "hard": truthy_junk}))
    assert result["archived"] == 1
    rows = provider_on_fake._graph.call("query_memories", cypher="RETURN 1")
    assert rows[0]["nodes"] == 1, "the node must still exist, archived rather than deleted"
    assert node_id is not None


@pytest.mark.parametrize("true_ish", [True, "true", "TRUE", " True "])
def test_hard_true_deletes(provider_on_fake, true_ish):
    _remember(provider_on_fake)
    result = json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "forget", "name": "Ava Walsh", "hard": true_ish}))
    assert result["archived"] == 1
    assert provider_on_fake._graph.call("query_memories", cypher="RETURN 1")[0]["nodes"] == 0


@pytest.mark.parametrize("junk", ["yes", "no", "0", "1", 0, 1, "", "TrUe!", [], {}, 1.0])
def test_ambiguous_booleans_are_refused_before_anything_is_deleted(provider_on_fake, junk):
    _remember(provider_on_fake)
    result = json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "forget", "name": "Ava Walsh", "hard": junk}))
    assert "must be true or false" in result["error"]
    assert provider_on_fake._graph.call("query_memories", cypher="RETURN 1")[0]["nodes"] == 1


@pytest.mark.parametrize("falsey, expected", [("false", False), (False, False), ("true", True), (True, True)])
def test_dry_run_strings_are_read_as_written(provider_on_fake, falsey, expected):
    result = json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "dream", "dry_run": falsey}))
    assert result["dream"]["dry_run"] is expected


@pytest.mark.parametrize("junk", ["yes", 1, "0", ""])
def test_dream_refuses_an_ambiguous_dry_run(provider_on_fake, junk):
    result = json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "dream", "dry_run": junk}))
    assert "must be true or false" in result["error"]


def test_a_bad_boolean_is_refused_whichever_action_carries_it(provider_on_fake):
    """A flag is validated on every action, so it cannot be ignored here and honoured there."""
    result = json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "stats", "hard": "maybe"}))
    assert "must be true or false" in result["error"]


def test_absent_booleans_default_to_false(provider_on_fake):
    _remember(provider_on_fake)
    assert json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "forget", "name": "Ava Walsh"}))["archived"] == 1
    assert provider_on_fake._graph.call("query_memories", cypher="RETURN 1")[0]["nodes"] == 1
    assert json.loads(provider_on_fake.handle_tool_call(
        "reverie", {"action": "dream"}))["dream"]["dry_run"] is False


def test_as_bool_covers_every_boolean_the_schema_advertises(reverie):
    schema = reverie.ReverieMemoryProvider(config={}).get_tool_schemas()[0]
    booleans = {name for name, spec in schema["parameters"]["properties"].items()
                if spec.get("type") == "boolean"}
    assert booleans == set(reverie.BOOLEAN_ARGS)


def test_negative_or_zero_limit_is_refused(reverie, client_factory):
    provider = reverie.ReverieMemoryProvider(config={})
    provider._graph = Graph(client_factory())
    for bad in (-1, 0):
        out = json.loads(provider.handle_tool_call("reverie", {"action": "search", "query": "x", "limit": bad}))
        assert "limit must be a positive integer" in out.get("error", "")
