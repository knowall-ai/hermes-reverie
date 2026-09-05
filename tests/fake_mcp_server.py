#!/usr/bin/env python3
"""A stand-in for the mcp-reverie MCP server, for tests.

It speaks the same stdio JSON-RPC 2.0 dialect (``initialize`` → ``notifications/initialized`` →
``tools/list`` / ``tools/call``) and implements the same tool surface over a tiny in-memory
graph, returning results in the same shapes: memories carry ``_id``/``_labels`` (plus
``_score``/``_match`` from a search), relationships carry ``_type``, deletes return
``deletedCount``. Embedding fields are never returned, exactly as the real server scrubs them.

Behaviour is steered by the environment so tests can provoke failures:

``FAKE_MCP_STATE``       JSON file the graph is persisted to, so a restarted server keeps it.
``FAKE_MCP_CRASH_AFTER`` exit hard after this many ``tools/call`` requests (crash/restart test).
``FAKE_MCP_TOOL_ERROR``  return ``isError`` for calls to this tool name.
``FAKE_MCP_HANG``        never answer ``tools/call`` (timeout test).
``FAKE_MCP_NO_START``    exit 1 immediately, after a line on stderr (startup-failure test).
``FAKE_MCP_ECHO_ENV``    comma-separated env names echoed back by the ``echo_env`` tool.
``FAKE_MCP_CALL_LOG``    file each ``tools/call`` appends its tool name to, before doing anything
                         else — so a test can count what actually reached the server even when
                         the call never answers.
"""
import json
import os
import re
import sys
import time

PROTOCOL_VERSION = "2024-11-05"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# The argument contract the real server enforces (mcp-reverie src/types.ts @ 01dceae):
# allowed top-level keys per tool, and nothing else.
ALLOWED_KEYS = {
    "search_memories": ("query", "label", "depth", "order_by", "limit", "since_date",
                        "search_mode", "similarity_threshold"),
    "create_memory": ("label", "properties"),
    "update_memory": ("nodeId", "properties"),
    "create_connection": ("fromMemoryId", "toMemoryId", "type", "properties"),
    "delete_memory": ("nodeId",),
    "delete_connection": ("fromMemoryId", "toMemoryId", "type"),
    "query_memories": ("cypher", "params"),
    "memory_stats": (),
    "dream": ("dry_run",),
    "echo_env": (),
}
HOUSEKEEPING = {"created_at", "updated_at", "status", "archived_at",
                "embedding", "name_embedding", "embedding_model", "embedded_at"}


class FakeGraph:
    def __init__(self, path=None):
        self.path = path
        self.nodes = {}   # id -> {"label": str, "props": {}}
        self.rels = []    # {"from": id, "to": id, "type": str, "props": {}}
        self.next_id = 1
        self.load()

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.nodes = {int(k): v for k, v in data.get("nodes", {}).items()}
        self.rels = data.get("rels", [])
        self.next_id = data.get("next_id", max(self.nodes, default=0) + 1)

    def save(self):
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"nodes": self.nodes, "rels": self.rels, "next_id": self.next_id}, handle)

    def create(self, label, props):
        node_id = self.next_id
        self.next_id += 1
        self.nodes[node_id] = {"label": label, "props": dict(props)}
        self.save()
        return node_id

    def memory(self, node_id, extra=None):
        node = self.nodes[node_id]
        payload = {k: v for k, v in node["props"].items() if k not in ("embedding", "name_embedding")}
        payload["_id"] = node_id
        payload["_labels"] = [node["label"]]
        payload.update(extra or {})
        return payload


def node_id(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid {what}: {value!r}")
    return value


def identifier(value, what):
    if not isinstance(value, str) or not IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid Cypher identifier for {what}: {value!r}")
    return value


def plain_object(value, what, required=True):
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {what}: expected an object")
    return value


def check_args(tool, args):
    """Reject unknown top-level keys and malformed values, exactly as the real server does."""
    allowed = ALLOWED_KEYS.get(tool)
    if allowed is None:
        return
    unknown = sorted(set(args) - set(allowed))
    if unknown:
        raise ValueError(f"Invalid {tool} arguments: unknown key(s) {', '.join(unknown)}")
    if tool == "search_memories":
        if "depth" in args and not (isinstance(args["depth"], int) and 0 <= args["depth"] <= 5):
            raise ValueError("Invalid search_memories arguments: depth")
        if "limit" in args and not (isinstance(args["limit"], int) and 1 <= args["limit"] <= 200):
            raise ValueError("Invalid search_memories arguments: limit")
        if "search_mode" in args and args["search_mode"] not in ("hybrid", "keyword", "semantic"):
            raise ValueError("Invalid search_memories arguments: search_mode")
        if "similarity_threshold" in args:
            threshold = args["similarity_threshold"]
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
                raise ValueError("Invalid search_memories arguments: similarity_threshold")
    if tool == "create_memory":
        identifier(args.get("label"), "label")
        plain_object(args.get("properties"), "properties")
    if tool == "update_memory":
        node_id(args.get("nodeId"), "nodeId")
        plain_object(args.get("properties"), "properties")
    if tool in ("create_connection", "delete_connection"):
        node_id(args.get("fromMemoryId"), "fromMemoryId")
        node_id(args.get("toMemoryId"), "toMemoryId")
        identifier(args.get("type"), "type")
        plain_object(args.get("properties"), "properties", required=False)
    if tool == "delete_memory":
        node_id(args.get("nodeId"), "nodeId")
    if tool == "query_memories":
        if not isinstance(args.get("cypher"), str) or not args["cypher"].strip():
            raise ValueError("Invalid query_memories arguments: cypher")
        plain_object(args.get("params"), "params", required=False)


def content_values(props):
    return [v for k, v in props.items() if not k.startswith("_") and k not in HOUSEKEEPING and v is not None]


def keyword_match(query, props):
    words = [w for w in query.strip().lower().split() if w]
    if not words:
        return True
    for value in content_values(props):
        haystack = " ".join(str(v) for v in value).lower() if isinstance(value, list) else str(value).lower()
        if any(word in haystack for word in words):
            return True
    return False


class FakeServer:
    def __init__(self):
        self.graph = FakeGraph(os.environ.get("FAKE_MCP_STATE"))
        self.calls = 0
        self.crash_after = int(os.environ.get("FAKE_MCP_CRASH_AFTER", "0") or 0)

    # -- tools -------------------------------------------------------------
    def search_memories(self, args):
        query = args.get("query", "") or ""
        label = args.get("label")
        depth = args.get("depth", 1)
        limit = min(int(args.get("limit", 10)), 200)
        mode = args.get("search_mode", "hybrid")
        rows = []
        for node_id, node in sorted(self.graph.nodes.items()):
            if label and node["label"].lower() != str(label).lower():
                continue
            if not keyword_match(query, node["props"]):
                continue
            memory = self.graph.memory(node_id, {"_score": 1.0, "_match": "keyword" if mode != "semantic" else "semantic"})
            connections = []
            if depth:
                for rel in self.graph.rels:
                    other = rel["to"] if rel["from"] == node_id else (rel["from"] if rel["to"] == node_id else None)
                    if other is None or other not in self.graph.nodes:
                        continue
                    connections.append({
                        "memory": self.graph.memory(other),
                        "relationship": {**rel["props"], "_id": 900 + self.graph.rels.index(rel), "_type": rel["type"]},
                        "distance": 1,
                    })
            rows.append({"memory": memory, "connections": connections})
        return rows[:limit]

    def create_memory(self, args):
        props = dict(args.get("properties") or {})
        props.setdefault("created_at", "2026-01-01T00:00:00Z")
        node_id = self.graph.create(args["label"], props)
        return {"memory": self.graph.memory(node_id)}

    def update_memory(self, args):
        node_id = int(args["nodeId"])
        if node_id not in self.graph.nodes:
            raise KeyError(f"no memory with id {node_id}")
        self.graph.nodes[node_id]["props"].update(args.get("properties") or {})
        self.graph.save()
        extra = {}
        real = [k for k in self.graph.nodes[node_id]["props"] if k not in HOUSEKEEPING and not k.startswith("_")]
        if len(real) > int(os.environ.get("FAKE_MCP_MAX_PROPERTIES", "30")):
            extra["_hint"] = f"{len(real)} properties is too many"
        return {"memory": self.graph.memory(node_id, extra)}

    def create_connection(self, args):
        rel = {"from": int(args["fromMemoryId"]), "to": int(args["toMemoryId"]),
               "type": args["type"], "props": dict(args.get("properties") or {})}
        self.graph.rels.append(rel)
        self.graph.save()
        return {"relationship": {**rel["props"], "_id": 900 + len(self.graph.rels), "_type": rel["type"]}}

    def delete_memory(self, args):
        node_id = int(args["nodeId"])
        existed = self.graph.nodes.pop(node_id, None) is not None
        self.graph.rels = [r for r in self.graph.rels if node_id not in (r["from"], r["to"])]
        self.graph.save()
        return {"deletedCount": 1 if existed else 0}

    def delete_connection(self, args):
        before = len(self.graph.rels)
        self.graph.rels = [
            r for r in self.graph.rels
            if not (r["from"] == int(args["fromMemoryId"]) and r["to"] == int(args["toMemoryId"])
                    and r["type"] == args["type"])
        ]
        self.graph.save()
        return {"deletedCount": before - len(self.graph.rels)}

    def query_memories(self, args):
        return [{"cypher": args.get("cypher"), "params": args.get("params") or {},
                 "nodes": len(self.graph.nodes)}]

    def memory_stats(self, _args):
        labels = {}
        for node in self.graph.nodes.values():
            if str(node["props"].get("status", "")).lower() == "archived":
                continue
            labels[node["label"]] = labels.get(node["label"], 0) + 1
        return {"nodes": sum(labels.values()), "relationships": len(self.graph.rels), "labels": labels,
                "relationship_types": {r["type"]: 1 for r in self.graph.rels},
                "embedded": 0, "embedder": None, "orphans": 0}

    def dream(self, args):
        return {"dry_run": bool(args.get("dry_run")), "relabelled": 0, "merged": 0, "reembedded": 0,
                "orphans": 0, "duplicates": [], "bloated": [], "apoc_available": True, "notes": []}

    def echo_env(self, _args):
        names = [n for n in (os.environ.get("FAKE_MCP_ECHO_ENV", "") or "").split(",") if n]
        return {name: os.environ.get(name) for name in names}

    TOOLS = ("search_memories", "create_memory", "update_memory", "create_connection", "delete_memory",
             "delete_connection", "query_memories", "memory_stats", "dream", "echo_env")

    # -- protocol ----------------------------------------------------------
    def call_tool(self, name, args):
        self.calls += 1
        log = os.environ.get("FAKE_MCP_CALL_LOG")
        if log:
            # Written first, so a hanging or crashing call still counts as having arrived.
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(name + "\n")
                handle.flush()
        if os.environ.get("FAKE_MCP_HANG"):
            time.sleep(300)  # never answers; the client must time out
        if self.crash_after and self.calls > self.crash_after:
            os._exit(9)
        if os.environ.get("FAKE_MCP_TOOL_ERROR") == name:
            return {"content": [{"type": "text", "text": f"{name} exploded"}], "isError": True}
        handler = getattr(self, name, None)
        if handler is None or name not in self.TOOLS:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
        try:
            check_args(name, args)
            payload = handler(args)
        except Exception as exc:  # the real server reports tool failures the same way
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}

    def handle(self, message):
        method = message.get("method")
        params = message.get("params") or {}
        if method == "initialize":
            return {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-reverie", "version": "0.0.0"}}
        if method == "tools/list":
            return {"tools": [{"name": name, "description": name,
                               "inputSchema": {"type": "object", "properties": {}}} for name in self.TOOLS]}
        if method == "tools/call":
            return self.call_tool(params.get("name"), params.get("arguments") or {})
        raise LookupError(f"unknown method {method}")

    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            if "id" not in message:  # a notification, e.g. notifications/initialized
                continue
            try:
                result = self.handle(message)
                response = {"jsonrpc": "2.0", "id": message["id"], "result": result}
            except Exception as exc:
                response = {"jsonrpc": "2.0", "id": message["id"],
                            "error": {"code": -32601, "message": str(exc)}}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def main():
    if os.environ.get("FAKE_MCP_NO_START"):
        sys.stderr.write("fake server refusing to start\n")
        sys.stderr.flush()
        sys.exit(1)
    sys.stderr.write("fake reverie MCP server running on stdio\n")
    sys.stderr.flush()
    FakeServer().run()


if __name__ == "__main__":
    main()
