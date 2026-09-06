"""Reverie's graph, over the mcp-reverie MCP server.

There is no Neo4j driver here any more. Every read and write goes through the
`mcp-reverie <https://github.com/knowall-ai/mcp-reverie>`_ server (binary ``reverie``, alias
``mcp-neo4j-agent-memory``) over stdio MCP, so the Hermes plugin and the MCP server share one
graph, one set of conventions, one hybrid keyword+semantic search and one Dreaming
implementation. This module is a thin adapter: it maps Reverie's verbs onto the server's tools
and normalises the answers back into the shapes the provider has always rendered.

  recall/search → ``search_memories``      remember → ``search_memories`` + ``create_memory``/``update_memory``
  connect       → ``create_connection``    forget   → ``update_memory`` (archive) or ``delete_memory``
  probe/cypher  → ``search_memories``/``query_memories``
  stats         → ``memory_stats``         dream    → ``dream``

Graph conventions (shared with the MCP server so several agents can read one another's data):
labels are capitalised singular (Person, Organization, Project, Product, Concept, Meeting,
Decision, Risk); every node has ``name``; matching is case-insensitive; relationships are
UPPER_SNAKE (WORKS_AT, HAS_ROLE, PARTNERS_WITH, INTRODUCED_BY, INTERESTED_IN, MET_WITH,
DISCUSSED, DECIDED, OWNS, BLOCKED_BY).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from .mcp_client import MCPClient, MCPError, MCPToolError, server_env_from_environ

logger = logging.getLogger(__name__)

#: Mirrors the server's IDENTIFIER_RE (mcp-reverie src/types.ts, contract 01dceae): labels and
#: relationship types are interpolated into Cypher, so they must be plain identifiers.
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
CANONICAL_LABELS = ("Person", "Organization", "Project", "Product", "Concept", "Meeting", "Decision", "Risk", "Pet")
#: Mirrors the server's read-only guard (src/cypher-guard.ts).
WRITE_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE)
CALL_RE = re.compile(r"\bCALL\b", re.IGNORECASE)
PROCEDURE_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)")
COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
READ_ONLY_PROCEDURES = frozenset({
    "db.labels", "db.relationshiptypes", "db.propertykeys",
    "db.schema.visualization", "db.schema.nodetypeproperties", "db.schema.reltypeproperties",
    "db.index.vector.querynodes", "db.index.vector.queryrelationships",
    "db.index.fulltext.querynodes", "db.index.fulltext.queryrelationships",
})

DEFAULT_SERVER_COMMAND = "reverie"
#: Search modes the server accepts (mcp-reverie ``src/search.ts`` @ 95baf2a). ``exact`` is
#: case-insensitive equality on ``name``, ``aliases`` or ``email`` — a lookup, not a ranking.
SEARCH_MODES = ("hybrid", "keyword", "semantic", "exact")
#: The modes that make sense as the *recall* default: exact belongs to name lookups, not to
#: reading a prompt, so it is not offered as a configured default.
RECALL_MODES = ("hybrid", "keyword", "semantic")
SEARCH_MAX_LIMIT = 200
SEARCH_MAX_DEPTH = 5
#: Embedding providers the server accepts in REVERIE_EMBEDDINGS.
EMBEDDING_PROVIDERS = ("local", "openai", "azure", "ollama", "voyage", "none")
#: Values Neo4j can store on a node or relationship.
PRIMITIVES = (str, int, float, bool)
#: How an exact-name lookup asks the server: ``search_mode: "exact"`` returns only memories whose
#: name, alias or email equals the query, case-insensitively, and archived memories are left out
#: server-side. The limit stays the server's maximum so every duplicate of a name is on the page —
#: :meth:`Graph.resolve` has to see all of them to refuse an ambiguous write. This dict is the one
#: switch point for how names are looked up.
NAME_SEARCH = {"search_mode": "exact", "limit": SEARCH_MAX_LIMIT, "depth": 0}
#: Neighbours rendered per recalled entity, as before.
MAX_RELS = 8


class MemoryNotFound(RuntimeError):
    """No memory carries this name (with this label, when one was given)."""

    def __init__(self, message: str, name: str, label: Optional[str] = None) -> None:
        super().__init__(message)
        self.name = name
        self.label = label


class AmbiguousMemory(RuntimeError):
    """Several memories carry this name, so the caller must say which one it meant."""

    def __init__(self, message: str, name: str, matches: List[Dict[str, Any]]) -> None:
        super().__init__(message)
        self.name = name
        self.matches = matches


def _ident(value: str, what: str) -> str:
    if not isinstance(value, str) or not IDENT_RE.match(value):
        raise ValueError(
            f"invalid {what}: {value!r} — must match [A-Za-z_][A-Za-z0-9_]{{0,63}} "
            "(labels and relationship types are identifiers, not free text)"
        )
    return value


def _node_id(value: Any, what: str = "node id") -> int:
    """The server takes non-negative integer node ids and rejects anything else."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {what}: {value!r} — must be a non-negative integer")
    return value


def _properties(props: Any, what: str) -> Dict[str, Any]:
    """A plain mapping of Neo4j-storable values: no arrays at the top level, no nested objects."""
    if props is None:
        return {}
    if not isinstance(props, dict):
        raise ValueError(f"{what} must be an object, not {type(props).__name__}")
    for key, value in props.items():
        if not isinstance(key, str):
            raise ValueError(f"{what} keys must be strings, got {key!r}")
        if value is None or isinstance(value, PRIMITIVES):
            continue
        if isinstance(value, (list, tuple)) and all(v is None or isinstance(v, PRIMITIVES) for v in value):
            continue
        raise ValueError(
            f"{what}['{key}'] is a {type(value).__name__}; a graph property must be text, a number, "
            "a boolean or a list of those — model anything richer as its own memory and connect it"
        )
    return dict(props)


def _int_in_range(value: Any, low: int, high: int, what: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {what}: {value!r} — must be an integer") from None
    return max(low, min(high, number))


def _threshold(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid similarity_threshold: {value!r} — must be a number in 0..1") from None
    if number != number or number in (float("inf"), float("-inf")) or not 0.0 <= number <= 1.0:
        raise ValueError(f"invalid similarity_threshold: {value!r} — must be a finite number in 0..1")
    return number


def read_only_violation(cypher: str) -> Optional[str]:
    """Why this Cypher is not read-only, or None. Mirrors the server's guard, comments and all."""
    stripped = COMMENT_RE.sub(" ", cypher or "")
    write = WRITE_RE.search(stripped)
    if write:
        return f"{write.group(1).upper()} is not allowed"
    for match in CALL_RE.finditer(stripped):
        rest = stripped[match.end():]
        if rest.lstrip().startswith("{"):
            return "CALL subqueries are not allowed"
        procedure = PROCEDURE_RE.match(rest)
        if not procedure:
            return "CALL could not be resolved to a procedure"
        if procedure.group(1).lower() not in READ_ONLY_PROCEDURES:
            return f"procedure {procedure.group(1)} is not on the read-only allow-list"
    return None


def canonical_label(label: str) -> str:
    """Map 'person' / 'PERSON' / 'organisation' to the canonical label; unknown labels are title-cased."""
    if not label:
        return "Concept"
    low = label.strip().lower()
    if low in ("organisation", "org", "company", "companies"):
        return "Organization"
    for canon in CANONICAL_LABELS:
        if low == canon.lower():
            return canon
    return _ident(label.strip()[:1].upper() + label.strip()[1:], "label")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _node_props(memory: Dict[str, Any]) -> Dict[str, Any]:
    """A server memory minus its ``_id``/``_labels``/``_score`` metadata."""
    return {k: v for k, v in (memory or {}).items() if not k.startswith("_")}


def _node_label(memory: Dict[str, Any]) -> str:
    labels = (memory or {}).get("_labels") or []
    return labels[0] if labels else "Memory"


class Graph:
    """Reverie's graph verbs, each one or two MCP tool calls.

    Args:
        client: a started-on-demand :class:`~reverie.mcp_client.MCPClient`.
        search_mode: default mode for recall (``hybrid``, ``keyword`` or ``semantic``).
        similarity_threshold: semantic cut-off passed to the server (it defaults to 0.4).
        depth: relationship depth included with each recalled entity (0-5).

    Archived memories (``status = 'archived'``) are excluded by the server, from results and from
    the neighbourhoods it returns with them; nothing here filters them again. No verb needs them
    back, so nothing passes ``include_archived: true`` — :meth:`search_memories` takes the flag for
    a caller that one day does.
    """

    def __init__(
        self,
        client: MCPClient,
        search_mode: str = "hybrid",
        similarity_threshold: Optional[float] = None,
        depth: int = 1,
    ) -> None:
        self.client = client
        if search_mode not in RECALL_MODES:
            raise ValueError(f"invalid search_mode: {search_mode!r} — one of {', '.join(RECALL_MODES)}")
        self.search_mode = search_mode
        self.similarity_threshold = similarity_threshold
        self.depth = max(0, min(5, int(depth)))

    @classmethod
    def from_env(cls, config: Optional[Dict[str, Any]] = None) -> "Graph":
        """Build a graph from plugin config plus the NEO4J_*/REVERIE_* environment.

        Config keys (all optional): ``server_command``, ``server_timeout``,
        ``server_startup_timeout``, ``embeddings``, ``model_cache``, ``search_mode``,
        ``similarity_threshold``, ``recall_depth``.
        """
        config = config or {}
        extra_env: Dict[str, Any] = {}
        embeddings = config.get("embeddings")
        if embeddings:
            if str(embeddings).lower() not in EMBEDDING_PROVIDERS:
                raise ValueError(
                    f"invalid embeddings provider: {embeddings!r} — one of {', '.join(EMBEDDING_PROVIDERS)}")
            extra_env["REVERIE_EMBEDDINGS"] = str(embeddings).lower()
        if config.get("model_cache"):
            extra_env["REVERIE_MODEL_CACHE"] = config["model_cache"]
        client = MCPClient(
            command=config.get("server_command") or DEFAULT_SERVER_COMMAND,
            env=server_env_from_environ(extra_env),
            timeout=float(config.get("server_timeout", 30.0)),
            startup_timeout=float(config.get("server_startup_timeout", 60.0)),
        )
        # Validate here, once, rather than failing the same way on every recall.
        mode = str(config.get("search_mode") or "hybrid")
        if mode not in RECALL_MODES:
            raise ValueError(f"invalid search_mode: {mode!r} — one of {', '.join(RECALL_MODES)}")
        return cls(
            client,
            search_mode=mode,
            similarity_threshold=_threshold(config.get("similarity_threshold")),
            depth=_int_in_range(config.get("recall_depth", 1), 0, SEARCH_MAX_DEPTH, "recall_depth"),
        )

    def close(self) -> None:
        try:
            self.client.stop()
        except Exception:
            pass

    def ping(self) -> bool:
        return self.client.ping()

    # -- low level ---------------------------------------------------------
    def call(self, tool: str, **arguments: Any) -> Any:
        """One tool call, with None arguments dropped (the server's schemas are strict)."""
        payload = {k: v for k, v in arguments.items() if v is not None}
        return self.client.call_tool(tool, payload)

    # -- recall ------------------------------------------------------------
    def search_memories(
        self,
        query: str,
        limit: int = 10,
        label: Optional[str] = None,
        depth: Optional[int] = None,
        search_mode: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        since_date: Optional[str] = None,
        order_by: Optional[str] = None,
        include_archived: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Raw ``search_memories`` rows: ``{memory: {...}, connections: [...]}``.

        Arguments are validated here, to the same rules the server enforces, so a bad call fails
        with a message Hermes can act on instead of a rejected tool call.

        ``include_archived`` is the only way to see soft-deleted memories: without it the server
        leaves ``status = 'archived'`` nodes out of the results *and* out of the connections it
        returns with the ones it keeps.
        """
        mode = search_mode or self.search_mode
        if mode not in SEARCH_MODES:
            raise ValueError(f"invalid search_mode: {mode!r} — one of {', '.join(SEARCH_MODES)}")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"invalid label: {label!r} — must be text")
        if include_archived is not None and not isinstance(include_archived, bool):
            raise ValueError(f"invalid include_archived: {include_archived!r} — must be true or false")
        rows = self.call(
            "search_memories",
            query=str(query or ""),
            label=label or None,  # the server matches the label case-insensitively
            depth=self.depth if depth is None else _int_in_range(depth, 0, SEARCH_MAX_DEPTH, "depth"),
            limit=_int_in_range(limit, 1, SEARCH_MAX_LIMIT, "limit"),
            search_mode=mode,
            similarity_threshold=_threshold(
                self.similarity_threshold if similarity_threshold is None else similarity_threshold
            ),
            since_date=since_date,
            order_by=order_by,
            include_archived=include_archived,
        )
        return rows if isinstance(rows, list) else []

    def recall(
        self,
        terms: Iterable[str],
        limit: int = 5,
        label: Optional[str] = None,
        depth: Optional[int] = None,
        search_mode: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Entities matching any of ``terms``, with their neighbours.

        The terms are joined into one query: the server's keyword pass matches a node when any
        word of the query appears in any of its content properties, which is the OR the old
        Cypher did, in a single round trip. In hybrid mode the semantic pass then adds close
        variants ("Ben Weeks" finding "Benjamin Weeks") that keywords miss. Archived nodes never
        come back — the server excludes them, from the hits and from their neighbourhoods — so the
        page asked for is the page used.
        """
        terms = [str(t).strip() for t in terms if t and str(t).strip()]
        if not terms:
            return []
        rows = self.search_memories(
            " ".join(terms),
            limit=_int_in_range(limit, 1, SEARCH_MAX_LIMIT, "limit"),
            label=label,
            depth=depth,
            search_mode=search_mode,
            similarity_threshold=similarity_threshold,
        )
        hits: List[Dict[str, Any]] = []
        seen: set = set()
        for hit in (self._to_hit(row) for row in rows):
            if hit is None or hit["id"] in seen:
                continue
            seen.add(hit["id"])
            hits.append(hit)
            if len(hits) >= int(limit):
                break
        return hits

    @staticmethod
    def _to_hit(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """One search row in the shape the provider renders: ``{id, label, props, rels, score, match}``.

        ``rels`` keeps the old key names. Direction is not recoverable from the server's
        scrubbed relationship objects (they carry properties, ``_id`` and ``_type`` only), so
        ``out`` is always True and neighbours read as "TYPE Other".
        """
        if not isinstance(row, dict):
            return None
        memory = row.get("memory")
        if not isinstance(memory, dict):
            return None
        rels: List[Dict[str, Any]] = []
        for conn in row.get("connections") or []:
            if not isinstance(conn, dict):
                continue
            other = conn.get("memory")
            rel = conn.get("relationship")
            if not isinstance(other, dict) or not isinstance(rel, dict):
                continue
            rels.append({
                "type": rel.get("_type"),
                "out": True,
                "name": other.get("name"),
                "label": _node_label(other),
                "distance": conn.get("distance"),
            })
        return {
            "id": memory.get("_id"),
            "label": _node_label(memory),
            "props": _node_props(memory),
            "rels": rels[:MAX_RELS],
            "score": memory.get("_score"),
            "match": memory.get("_match"),
        }

    def probe(self, name: str) -> List[Dict[str, Any]]:
        return self.recall([name.strip()], limit=3)

    def find_all(self, name: str, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every live node whose ``name`` — or alias, or email — equals ``name``, case-insensitively.

        This is the dedupe step in front of every write: the graph must never grow a second
        "Atlantic Pharma" because the agent typed it differently. The whole match runs on the
        server (``search_mode: "exact"``), so "James Kelly" cannot be crowded out of the page by
        the hundred nodes a keyword search for "James" would return, and an alias or email the
        agent typed instead of the name still finds the node it belongs to. The label, when given,
        is pushed down too: it narrows what "the node called X" means when several labels share a
        name. Archived nodes are excluded server-side.
        """
        wanted = (name or "").strip()
        if not wanted:
            return []
        rows = self.search_memories(wanted, label=label, **NAME_SEARCH)
        return [hit for hit in (self._to_hit(row) for row in rows) if hit is not None]

    def find(self, name: str, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The one node whose name, alias or email equals ``name`` case-insensitively, or None.

        Several matches mean the graph holds duplicates (Dreaming merges them); the first is
        used, as it always was. Use :meth:`find_all` where ambiguity must be refused instead.
        """
        matches = self.find_all(name, label=label)
        if len(matches) > 1:
            logger.debug("Reverie: %r matches %d memories; using the first", name, len(matches))
        return matches[0] if matches else None

    def resolve(self, name: str, label: Optional[str] = None, what: str = "entity") -> Dict[str, Any]:
        """The one node called ``name``, or a refusal that says what to do about it.

        Writing a relationship to the wrong "Ava" is worse than not writing it, so ambiguity is
        an error here rather than a guess. The two refusals are distinct exception types —
        :class:`MemoryNotFound` and :class:`AmbiguousMemory` — so callers branch on the type,
        never on the wording of a message that carries a caller-supplied name.
        """
        matches = self.find_all(name, label=label)
        if not matches:
            where = f" with label {str(label).strip()}" if label else ""
            raise MemoryNotFound(f"{what} {name!r}{where} not found — remember it first", name, label)
        if len(matches) > 1:
            labels = sorted({hit["label"] for hit in matches})
            raise AmbiguousMemory(
                f"{what} {name!r} is ambiguous: {len(matches)} memories share that name"
                + (f" ({', '.join(labels)})" if len(labels) > 1 else "")
                + " — say which by passing a label, or merge them with dream",
                name, matches,
            )
        return matches[0]

    # -- writes ------------------------------------------------------------
    def remember(self, label: str, name: str, props: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create or update one entity, deduped by exact (case-insensitive) name within the label."""
        label = canonical_label(label)
        name = (name or "").strip()
        if not name:
            raise ValueError("remember needs a name")
        props = _properties(props, "properties")
        props = {k: v for k, v in props.items() if k != "name" and v is not None}

        existing = self.find(name, label=label)
        if existing is not None:
            result = self.call("update_memory", nodeId=_node_id(existing["id"]),
                               properties={**props, "updated_at": _now_iso()})
        else:
            result = self.call("create_memory", label=_ident(label, "label"),
                               properties={"name": name, **props})
        memory = (result or {}).get("memory") if isinstance(result, dict) else None
        if not isinstance(memory, dict):
            raise RuntimeError(f"remember: unexpected server response: {result!r}")
        node = {"id": memory.get("_id"), "label": _node_label(memory), "props": _node_props(memory)}
        if memory.get("_hint"):
            node["hint"] = memory["_hint"]  # the server flags a node that has grown too many properties
        return node

    def connect(self, from_name: str, to_name: str, rel_type: str,
                props: Optional[Dict[str, Any]] = None,
                from_label: Optional[str] = None, to_label: Optional[str] = None) -> Dict[str, Any]:
        """Relate two remembered entities.

        ``from_label`` / ``to_label`` disambiguate an end when several memories share a name — a
        Person and a Project both called "Atlas", say. Without them, an ambiguous name is refused
        rather than guessed.
        """
        rel_type = _ident(rel_type.strip().upper(), "relationship type")
        source = self.resolve(from_name, from_label, "connect: source")
        target = self.resolve(to_name, to_label, "connect: target")
        self.call(
            "create_connection",
            fromMemoryId=_node_id(source["id"], "fromMemoryId"),
            toMemoryId=_node_id(target["id"], "toMemoryId"),
            type=rel_type,
            properties={**_properties(props, "properties"), "created_at": _now_iso()},
        )
        return {
            "from": source["props"].get("name"), "type": rel_type, "to": target["props"].get("name"),
            "props": props or {},
        }

    def forget(self, name: str, hard: bool = False) -> int:
        """Archive an entity — or delete it outright with ``hard=True``.

        Archiving stays the default so ``forget`` keeps the behaviour SOUL and the skills expect:
        the node stops being recalled — the server leaves ``status = 'archived'`` out of every
        search and out of the neighbourhoods of the nodes it does return — but its history and
        relationships survive. ``hard`` maps to the server's ``delete_memory``, which detaches
        and deletes.
        """
        hit = self.find(name)
        if hit is None:
            return 0
        if hard:
            result = self.call("delete_memory", nodeId=_node_id(hit["id"]))
            return int((result or {}).get("deletedCount", 0)) if isinstance(result, dict) else 0
        self.call(
            "update_memory", nodeId=_node_id(hit["id"]),
            properties={"status": "archived", "archived_at": _now_iso()},
        )
        return 1

    def forget_connection(self, from_name: str, to_name: str, rel_type: str,
                          from_label: Optional[str] = None, to_label: Optional[str] = None) -> int:
        """Delete one relationship (the server has no soft delete for relationships).

        Both ends take an optional label, for the same reason ``connect`` does: deleting the
        wrong "Atlas" edge is not undoable. An end that matches nothing is a no-op; an end that
        matches several memories is refused.
        """
        rel_type = _ident(rel_type.strip().upper(), "relationship type")
        try:
            source = self.resolve(from_name, from_label, "forget: source")
            target = self.resolve(to_name, to_label, "forget: target")
        except MemoryNotFound:
            return 0  # nothing to disconnect; an ambiguous end still raises
        result = self.call(
            "delete_connection",
            fromMemoryId=_node_id(source["id"], "fromMemoryId"),
            toMemoryId=_node_id(target["id"], "toMemoryId"),
            type=rel_type,
        )
        return int((result or {}).get("deletedCount", 0)) if isinstance(result, dict) else 0

    # -- read-only Cypher --------------------------------------------------
    def read_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Read-only Cypher via ``query_memories``.

        The same guard the server applies (comments stripped first, write clauses refused, CALL
        only for the read-only ``db.*`` procedures) runs here so the agent gets the reason back
        without a round trip. The server still runs the query in a READ transaction, capped at
        200 rows and 10 seconds.
        """
        if not isinstance(cypher, str) or not cypher.strip():
            raise ValueError("cypher action needs a query")
        violation = read_only_violation(cypher)
        if violation:
            raise ValueError(f"cypher action is read-only: {violation}; use remember/connect/forget for writes")
        if params is not None and not isinstance(params, dict):
            raise ValueError("cypher params must be an object")
        rows = self.call("query_memories", cypher=cypher, params=params or None)
        return rows if isinstance(rows, list) else ([] if rows is None else [rows])

    # -- dreaming and stats ------------------------------------------------
    def dream(self, dry_run: bool = False) -> Dict[str, Any]:
        """Server-side hygiene: relabel lowercase labels, merge duplicates, re-embed, report bloat.

        The report carries ``relabelled``, ``merged``, ``reembedded``, ``orphans``,
        ``duplicates`` (each ``{label, name, keep, merged[], skipped[{id, reason}]}``),
        ``bloated`` (nodes past ``REVERIE_MAX_PROPERTIES``), ``apoc_available`` and ``notes``.
        """
        result = self.call("dream", dry_run=bool(dry_run))
        return result if isinstance(result, dict) else {"result": result}

    def stats(self) -> Dict[str, Any]:
        """Graph counts: ``nodes``, ``relationships``, ``labels``, ``relationship_types``, ``embedded``, ``orphans``."""
        result = self.call("memory_stats")
        return result if isinstance(result, dict) else {}


__all__ = [
    "AmbiguousMemory", "CANONICAL_LABELS", "DEFAULT_SERVER_COMMAND", "EMBEDDING_PROVIDERS",
    "Graph", "MCPClient", "MCPError", "MCPToolError", "MemoryNotFound", "NAME_SEARCH",
    "RECALL_MODES", "SEARCH_MODES", "canonical_label", "read_only_violation",
]
