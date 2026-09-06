"""Reverie — KnowAll graph memory that dreams (Hermes memory-provider plugin).

Recall: before every non-trivial turn the prompt's names, emails and quoted phrases are
matched against the graph and the hits, with their relationships, are injected as context.
Store: the agent writes through the ``reverie`` tool (remember / connect / forget) so the graph
only ever holds curated entities, never raw chat. Dreaming: the nightly consolidation skill
installed by ``post_setup`` reviews the day and tidies the graph.

The graph itself lives behind the `mcp-reverie <https://github.com/knowall-ai/mcp-reverie>`_
MCP server (binary ``reverie``), which this plugin spawns and talks to over stdio — see
``mcp_client.py`` and ``graph.py``. Nothing here speaks Bolt, so search (hybrid keyword +
semantic), Dreaming and the graph conventions are shared with every other Reverie client.

Config: ``plugins.reverie`` in config.yaml (recall_limit, server_command, embeddings,
search_mode, …); secrets in .env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE),
which are passed through to the server process.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt

from .graph import CANONICAL_LABELS, DEFAULT_SERVER_COMMAND, SEARCH_MODES, Graph

logger = logging.getLogger(__name__)

GLYPH = "🌙"
PLUGIN_DIR = Path(__file__).resolve().parent

# Words that look like names but never are.
_STOP = {
    "The", "This", "That", "These", "Those", "Please", "Thanks", "Hello", "Hi", "Yes", "No", "Ok", "Okay",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
    "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
    "Teams", "Session", "Meeting", "Email", "Subject", "From", "To", "Re", "Fw", "Fwd", "Today", "Tomorrow",
}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_QUOTED_RE = re.compile(r"[\"“']([^\"”']{3,60})[\"”']")
_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z&'.-]+(?:\s+[A-Z][a-zA-Z&'.-]+){0,3})\b")


def recall_terms(text: str, max_terms: int = 8) -> List[str]:
    """Candidate entity strings from a prompt: emails, quoted phrases, capitalised runs."""
    if not text:
        return []
    terms: List[str] = []
    seen = set()

    def add(t: str) -> None:
        t = t.strip(" .,;:!?")
        if len(t) < 3 or t.lower() in seen:
            return
        seen.add(t.lower())
        terms.append(t)

    for m in _EMAIL_RE.findall(text):
        add(m)
    for m in _QUOTED_RE.findall(text):
        add(m)
    for m in _PROPER_RE.findall(text):
        if m.split()[0] not in _STOP:
            add(m)
    return terms[:max_terms]


#: Boolean arguments the ``reverie`` tool accepts. Each one flips a destructive or
#: behaviour-changing switch, so each is validated rather than coerced.
BOOLEAN_ARGS = ("hard", "dry_run")


def _as_bool(value: Any, name: str, default: bool = False) -> bool:
    """A tool argument as a real boolean, or a refusal.

    ``bool(value)`` is not usable here: a model that emits JSON by hand writes ``"hard": "false"``
    often enough, and every non-empty string is truthy — so "false" would hard-delete the node.
    Real booleans pass; the strings "true"/"false" are accepted case-insensitively and with
    surrounding whitespace trimmed, because they are unambiguous; everything else ("yes", "0", 1,
    "") is refused, because guessing at
    what a caller meant is how a soft archive becomes a permanent delete.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(
        f"{name} must be true or false, not {value!r} — use a JSON boolean (or the string "
        '"true" / "false")'
    )


def _load_plugin_config() -> dict:
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        return cfg_get(load_config_readonly(), "plugins", "reverie", default={}) or {}
    except Exception:
        return {}


REVERIE_TOOL = {
    "name": "reverie",
    "description": (
        "Reverie: your knowledge graph of people, organisations, projects, products, concepts, "
        "meetings and decisions. Recall happens automatically before each turn; use this tool to "
        "look deeper or to write.\n\nACTIONS:\n"
        "• search — hybrid keyword + semantic search, so 'Ben Weeks' also finds 'Benjamin Weeks'. "
        "Optional search_mode (hybrid | keyword | semantic | exact) and similarity_threshold (0-1, "
        "default 0.4) tune it: exact to ask whether one name is already in the graph (it matches only a "
        "memory whose name, alias or email equals the query), keyword for a literal word, semantic for "
        "'who else is like this', a higher threshold for fewer, closer matches. Archived memories are "
        "never returned.\n"
        "• probe — everything about one entity and its relationships.\n"
        "• remember — create or update ONE entity: label + name + its own scalar properties (role, email, "
        "aliases, notes). Where someone works is a WORKS_AT edge to an Organization node, not a company property. An existing entity with the same name is updated, never duplicated.\n"
        "• connect — relate two remembered entities: from, to, type (WORKS_AT, HAS_ROLE, PARTNERS_WITH, "
        "INTRODUCED_BY, INTERESTED_IN, MET_WITH, DISCUSSED, DECIDED, OWNS, BLOCKED_BY, MARRIED_TO, "
        "CUSTOMER_OF, SUPPLIES) and properties. "
        "If a name matches more than one memory the call is refused, not guessed — say which with "
        "from_label / to_label.\n"
        "• forget — archive an entity by name (soft delete); with hard=true delete it outright. To "
        "delete one relationship instead, pass from, to and type together — half of them is refused, "
        "not treated as an entity delete.\n"
        "• cypher — read-only Cypher for anything else.\n"
        "• stats — node, relationship, label and orphan counts.\n"
        "• dream — run graph hygiene (merge duplicates, fix labels, refresh embeddings, report bloated "
        "nodes); dry_run=true reports without writing.\n\n"
        "Search before you remember: never create a person or organisation that already exists under "
        "another spelling. Labels are capitalised singular. Store facts, not chat.\n\n"
        "GRAPH SHAPE — one node per thing, one edge per relationship. Every person, pet, organisation, "
        "project, product, meeting and decision someone mentions is its own node, then connected: a "
        "spouse is a Person linked MARRIED_TO, a colleague a Person linked WORKS_AT their organisation, a "
        "cat a Pet linked OWNS from its owner, a former employee keeps the WORKS_AT edge with "
        "status: former. Properties hold only facts about that node itself (email, role, spelling, "
        "verified: true or false); never write another entity into a notes/pets/family/colleagues property — "
        "a name inside a property cannot be searched, connected or corrected. Test: if a fact names a "
        "second thing, it is a node and an edge, not a property. Name nodes the way a person would say "
        "them: a Meeting is 'Kick-off call with Acme — 2026-09-05', never the sentence used to arrange it; "
        "purpose and outcome go in a summary property."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "probe", "remember", "connect", "forget", "cypher", "stats", "dream"]},
            "query": {"type": "string", "description": "search text, or Cypher for 'cypher'"},
            "name": {"type": "string", "description": "entity name for probe/remember/forget"},
            "label": {"type": "string", "description": "Person | Organization | Project | Product | Concept | Meeting | Decision | Risk | Pet"},
            "properties": {"type": "object", "description": "properties to set on the entity or relationship"},
            "from": {"type": "string", "description": "connect/forget: source entity name"},
            "to": {"type": "string", "description": "connect/forget: target entity name"},
            "from_label": {"type": "string", "description": "connect/forget: label of the source, when the name alone is ambiguous"},
            "to_label": {"type": "string", "description": "connect/forget: label of the target, when the name alone is ambiguous"},
            "type": {"type": "string", "description": "connect/forget: relationship type in UPPER_SNAKE"},
            "limit": {"type": "integer", "description": "max results (default 10)", "minimum": 1},
            "search_mode": {"type": "string", "enum": list(SEARCH_MODES),
                            "description": "search: hybrid (default), keyword-only, semantic-only, or "
                                           "exact (case-insensitive equality on name, alias or email)"},
            "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1,
                                     "description": "search: semantic cut-off, 0-1 (server default 0.4)"},
            "depth": {"type": "integer", "minimum": 0, "maximum": 5,
                      "description": "search: relationship hops to include (default 1)"},
            "hard": {"type": "boolean", "description": "forget: delete instead of archiving (true or false, never a word like 'yes')"},
            "dry_run": {"type": "boolean", "description": "dream: report planned changes without writing (true or false)"},
        },
        "required": ["action"],
    },
}


class ReverieMemoryProvider(MemoryProvider):
    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_plugin_config()
        self._graph: Optional[Graph] = None
        self._last: Optional[RecallStatus] = None
        self._recall_limit = int(self._config.get("recall_limit", 5))

    # -- identity ----------------------------------------------------------
    @property
    def name(self) -> str:
        return "reverie"

    def _server_command(self) -> str:
        return str(self._config.get("server_command") or DEFAULT_SERVER_COMMAND)

    def _server_argv(self) -> List[str]:
        """The configured command as argv, or [] when it is unparseable (an unbalanced quote)."""
        try:
            return shlex.split(self._server_command())
        except ValueError as exc:
            logger.warning("Reverie: server_command %r cannot be parsed: %s",
                           self._server_command(), exc)
            return []

    def is_available(self) -> bool:
        command = self._server_argv()
        if not command or not shutil.which(command[0]):
            return False
        return bool(os.environ.get("NEO4J_PASSWORD"))

    def unavailable_reason(self) -> str:
        command = self._server_argv()
        if not command:
            return (f"Reverie's server_command is not a valid command line: "
                    f"{self._server_command()!r} — check the quoting in config.yaml")
        if not shutil.which(command[0]):
            return (f"Reverie needs the mcp-reverie server on PATH (looked for '{command[0]}'): "
                    "npm install -g github:knowall-ai/mcp-reverie")
        return "Reverie needs NEO4J_PASSWORD (plus NEO4J_URI if not bolt://127.0.0.1:7687) in .env"

    # -- setup -------------------------------------------------------------
    def get_config_schema(self):
        return [
            {"key": "NEO4J_URI", "description": "Bolt URI (passed to the MCP server)", "default": "bolt://127.0.0.1:7687", "secret": True, "env_var": "NEO4J_URI"},
            {"key": "NEO4J_USERNAME", "description": "Neo4j user", "default": "neo4j", "secret": True, "env_var": "NEO4J_USERNAME"},
            {"key": "NEO4J_PASSWORD", "description": "Neo4j password", "required": True, "secret": True, "env_var": "NEO4J_PASSWORD"},
            {"key": "server_command", "description": "mcp-reverie server command", "default": DEFAULT_SERVER_COMMAND},
            {"key": "embeddings", "description": "Embeddings for semantic search: local, openai, azure, ollama, voyage or none", "default": "local"},
            {"key": "search_mode", "description": "Recall mode: hybrid, keyword or semantic", "default": "hybrid"},
            {"key": "similarity_threshold", "description": "Semantic similarity cut-off, 0-1 (blank = the server's 0.4)", "default": ""},
            {"key": "recall_limit", "description": "Entities recalled per turn", "default": "5", "type": "integer", "minimum": 1, "maximum": 20},
            {"key": "recall_depth", "description": "Relationship hops recalled with each entity", "default": "1", "type": "integer", "minimum": 0, "maximum": 5},
            {"key": "server_timeout", "description": "Seconds to wait for a tool call", "default": "30"},
            {"key": "server_startup_timeout", "description": "Seconds to wait for the server handshake (the local embedding model downloads on first use)", "default": "60"},
            {"key": "dreaming_schedule", "description": "Cron schedule for nightly Dreaming (blank = don't install)", "default": "0 3 * * *"},
        ]

    CONFIG_KEYS = ("recall_limit", "dreaming_schedule", "server_command", "embeddings", "search_mode",
                   "similarity_threshold", "recall_depth", "server_timeout", "server_startup_timeout",
                   "model_cache")

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        keep = {k: v for k, v in values.items() if k in self.CONFIG_KEYS}
        try:
            import yaml
            from hermes_cli.config import read_user_config_raw
            path = Path(hermes_home) / "config.yaml"
            existing = read_user_config_raw(path)
            existing.setdefault("plugins", {})["reverie"] = keep
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as exc:
            logger.warning("Reverie: could not persist config: %s", exc)

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Install the Dreaming skill into the profile and schedule it (idempotent)."""
        home = Path(hermes_home)
        src = PLUGIN_DIR / "skills" / "dreaming"
        dst = home / "skills" / "dreaming"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Dreaming skill installed to {dst}")
        except Exception as exc:
            print(f"  Could not install Dreaming skill: {exc}")
        schedule = (config or {}).get("dreaming_schedule", self._config.get("dreaming_schedule", "0 3 * * *"))
        if not schedule:
            return
        try:
            listing = subprocess.run(["hermes", "cron", "list"], capture_output=True, text=True, timeout=30).stdout
            if "dreaming" in listing:
                print("  Dreaming cron job already present")
                return
            subprocess.run(
                ["hermes", "cron", "create", "--name", "dreaming", "--deliver", "local", "--skill", "dreaming",
                 schedule, "It is time to dream. Follow the dreaming skill end to end and write today's dream diary."],
                check=False, timeout=60,
            )
            print(f"  Dreaming scheduled: {schedule}")
        except Exception as exc:
            print(f"  Could not schedule Dreaming ({exc}); run: hermes cron create --name dreaming --skill dreaming '{schedule}' 'Dream.'")

    # -- lifecycle ---------------------------------------------------------
    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._context = kwargs.get("agent_context", "primary")
        try:
            self._graph = Graph.from_env(self._config)
            if not self._graph.ping():
                logger.warning("Reverie: the mcp-reverie server (%s) did not answer; is Neo4j up at %s?",
                               self._server_command(), os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
        except Exception as exc:
            logger.warning("Reverie: could not start the mcp-reverie server: %s", exc)
            self._graph = None

    def shutdown(self) -> None:
        if self._graph:
            self._graph.close()
            self._graph = None

    def system_prompt_block(self) -> str:
        if not self._graph:
            return ""
        try:
            total = int(self._graph.stats().get("nodes", 0))
        except Exception:
            total = 0
        head = f"{GLYPH} Reverie graph memory is active ({total} entities)." if total else \
            f"{GLYPH} Reverie graph memory is active and empty — remember the people, organisations and projects you meet."
        return (
            "# Reverie\n" + head + "\n"
            "Relevant entities are recalled automatically before each turn under '## Reverie recalls'. "
            "Use the reverie tool to probe deeper, and to remember/connect facts worth keeping: who works where, "
            "who decided what, which project is blocked by whom. Remember as soon as you hear a durable fact, "
            "not at the end. One node per person/pet/organisation/project and one edge per relationship — "
            "never a list of names inside a property. Search before creating. Labels: "
            + ", ".join(CANONICAL_LABELS) + " (Pet is accepted too)."
        )

    # -- recall ------------------------------------------------------------
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last = None
        if not self._graph or is_trivial_prompt(query):
            return ""
        terms = recall_terms(query)
        if not terms:
            return ""
        try:
            hits = self._graph.recall(terms, limit=self._recall_limit)
        except Exception as exc:
            logger.debug("Reverie recall failed: %s", exc)
            return ""
        if not hits:
            return ""
        self._last = RecallStatus(provider_label="Reverie", count=len(hits), glyph=GLYPH)
        return "## Reverie recalls\n" + "\n".join(self._format(h) for h in hits)

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last

    @staticmethod
    def _format(hit: Dict[str, Any]) -> str:
        p = hit.get("props") or {}
        bits = [b for b in (p.get("role"), p.get("company"), p.get("email")) if b]
        line = f"- {hit.get('label')} **{p.get('name')}**" + (f" ({', '.join(bits)})" if bits else "")
        if p.get("notes"):
            line += f": {str(p['notes'])[:160]}"
        rels = [r for r in (hit.get("rels") or []) if r.get("type")]
        if rels:
            line += " — " + "; ".join(
                (f"{r['type']} {r['name']}" if r.get("out") else f"{r['name']} {r['type']} →") for r in rels[:6]
            )
        return line

    # -- tools -------------------------------------------------------------
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REVERIE_TOOL]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "reverie":
            return json.dumps({"error": f"unknown tool {tool_name}"})
        if not self._graph:
            return json.dumps({"error": "Reverie is not connected to its graph server"})
        action = args.get("action")
        try:
            limit = int(args.get("limit", 10))
            if limit < 1:
                return json.dumps({"error": f"limit must be a positive integer, not {limit}"})
            for flag in BOOLEAN_ARGS:
                if flag in args:
                    _as_bool(args[flag], f"'{flag}'")
            if action == "search":
                results = self._graph.recall(
                    [args.get("query", "")], limit=limit, label=args.get("label"),
                    depth=args.get("depth"), search_mode=args.get("search_mode"),
                    similarity_threshold=args.get("similarity_threshold"),
                )
                return json.dumps({"results": results}, default=str)
            if action == "probe":
                return json.dumps({"results": self._graph.probe(args.get("name") or args.get("query", ""))}, default=str)
            if action == "remember":
                if not args.get("name"):
                    return json.dumps({"error": "remember needs 'name'"})
                node = self._graph.remember(args.get("label") or "Concept", args["name"], args.get("properties"))
                return json.dumps({"remembered": node}, default=str)
            if action == "connect":
                for k in ("from", "to", "type"):
                    if not args.get(k):
                        return json.dumps({"error": f"connect needs '{k}'"})
                connected = self._graph.connect(
                    args["from"], args["to"], args["type"], args.get("properties"),
                    from_label=args.get("from_label"), to_label=args.get("to_label"),
                )
                return json.dumps({"connected": connected}, default=str)
            if action == "forget":
                # 'from'/'to'/'type' mean "delete this relationship". Half of them is a mistake,
                # and falling through would archive an entity instead — say so rather than guess.
                edge = {k: args.get(k) for k in ("from", "to", "type")}
                given = [k for k, v in edge.items() if v]
                if given and len(given) < 3:
                    missing = [k for k in ("from", "to", "type") if k not in given]
                    return json.dumps({"error": (
                        "forget: to delete a relationship give 'from', 'to' and 'type' together "
                        f"(missing {', '.join(repr(m) for m in missing)}); to archive an entity "
                        "give 'name' on its own")})
                if given:
                    deleted = self._graph.forget_connection(
                        edge["from"], edge["to"], edge["type"],
                        from_label=args.get("from_label"), to_label=args.get("to_label"),
                    )
                    return json.dumps({"disconnected": deleted})
                if not args.get("name"):
                    return json.dumps({"error": "forget needs 'name', or 'from'+'to'+'type'"})
                hard = _as_bool(args.get("hard"), "forget: 'hard'")
                return json.dumps({"archived": self._graph.forget(args["name"], hard=hard)})
            if action == "cypher":
                return json.dumps({"rows": self._graph.read_cypher(args.get("query", ""), args.get("properties"))[:limit]}, default=str)
            if action == "stats":
                return json.dumps({"counts": self._graph.stats()}, default=str)
            if action == "dream":
                dry_run = _as_bool(args.get("dry_run"), "dream: 'dry_run'")
                return json.dumps({"dream": self._graph.dream(dry_run=dry_run)}, default=str)
            return json.dumps({"error": f"unknown action {action}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def backup_paths(self) -> List[str]:
        return []


def register(ctx) -> None:
    ctx.register_memory_provider(ReverieMemoryProvider())
