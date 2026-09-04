"""Reverie — KnowAll graph memory that dreams (Hermes memory-provider plugin).

Recall: before every non-trivial turn the prompt's names, emails and quoted phrases are
matched against the Neo4j graph and the hits, with their relationships, are injected as
context. Store: the agent writes through the ``reverie`` tool (remember / connect / forget)
so the graph only ever holds curated entities, never raw chat. Dreaming: the nightly
consolidation skill installed by ``post_setup`` reviews the day and tidies the graph.

Config: ``plugins.reverie`` in config.yaml (recall_limit, recall_labels); secrets in .env
(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt

from .graph import CANONICAL_LABELS, Graph, canonical_label

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
        "• search — find nodes by name/alias/email (case-insensitive).\n"
        "• probe — everything about one entity and its relationships.\n"
        "• remember — create or update an entity: label + name + properties (role, email, company, notes).\n"
        "• connect — relate two remembered entities: from, to, type (WORKS_AT, HAS_ROLE, PARTNERS_WITH, "
        "INTRODUCED_BY, INTERESTED_IN, MET_WITH, DISCUSSED, DECIDED, OWNS, BLOCKED_BY) and properties.\n"
        "• forget — archive an entity (soft delete).\n"
        "• cypher — read-only Cypher for anything else.\n"
        "• stats — node counts by label.\n"
        "• dream — run graph hygiene (merge case-duplicates, fix labels, count orphans).\n\n"
        "Search before you remember: never create a person or organisation that already exists under "
        "another spelling. Labels are capitalised singular. Store facts, not chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "probe", "remember", "connect", "forget", "cypher", "stats", "dream"]},
            "query": {"type": "string", "description": "search text, or Cypher for 'cypher'"},
            "name": {"type": "string", "description": "entity name for probe/remember/forget"},
            "label": {"type": "string", "description": "Person | Organization | Project | Product | Concept | Meeting | Decision | Risk"},
            "properties": {"type": "object", "description": "properties to set on the entity or relationship"},
            "from": {"type": "string", "description": "connect: source entity name"},
            "to": {"type": "string", "description": "connect: target entity name"},
            "type": {"type": "string", "description": "connect: relationship type in UPPER_SNAKE"},
            "limit": {"type": "integer", "description": "max results (default 10)"},
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

    def is_available(self) -> bool:
        try:
            import neo4j  # noqa: F401
        except Exception:
            return False
        return bool(os.environ.get("NEO4J_PASSWORD"))

    def unavailable_reason(self) -> str:
        return "Reverie needs the neo4j driver and NEO4J_PASSWORD (plus NEO4J_URI if not bolt://127.0.0.1:7687) in .env"

    # -- setup -------------------------------------------------------------
    def get_config_schema(self):
        return [
            {"key": "NEO4J_URI", "description": "Bolt URI", "default": "bolt://127.0.0.1:7687", "secret": True, "env_var": "NEO4J_URI"},
            {"key": "NEO4J_USERNAME", "description": "Neo4j user", "default": "neo4j", "secret": True, "env_var": "NEO4J_USERNAME"},
            {"key": "NEO4J_PASSWORD", "description": "Neo4j password", "required": True, "secret": True, "env_var": "NEO4J_PASSWORD"},
            {"key": "recall_limit", "description": "Entities recalled per turn", "default": "5", "type": "integer", "minimum": 1, "maximum": 20},
            {"key": "dreaming_schedule", "description": "Cron schedule for nightly Dreaming (blank = don't install)", "default": "0 3 * * *"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        keep = {k: v for k, v in values.items() if k in ("recall_limit", "dreaming_schedule")}
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
            self._graph = Graph.from_env()
            if not self._graph.ping():
                logger.warning("Reverie: Neo4j did not answer at %s", os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
        except Exception as exc:
            logger.warning("Reverie: could not connect to Neo4j: %s", exc)
            self._graph = None

    def shutdown(self) -> None:
        if self._graph:
            self._graph.close()
            self._graph = None

    def system_prompt_block(self) -> str:
        if not self._graph:
            return ""
        try:
            counts = self._graph.stats()
            total = sum(counts.values())
        except Exception:
            total = 0
        head = f"{GLYPH} Reverie graph memory is active ({total} entities)." if total else \
            f"{GLYPH} Reverie graph memory is active and empty — remember the people, organisations and projects you meet."
        return (
            "# Reverie\n" + head + "\n"
            "Relevant entities are recalled automatically before each turn under '## Reverie recalls'. "
            "Use the reverie tool to probe deeper, and to remember/connect facts worth keeping: who works where, "
            "who decided what, which project is blocked by whom. Search before creating. Labels: "
            + ", ".join(CANONICAL_LABELS) + "."
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
            return json.dumps({"error": "Reverie is not connected to Neo4j"})
        action = args.get("action")
        limit = int(args.get("limit", 10))
        try:
            if action == "search":
                return json.dumps({"results": self._graph.recall([args.get("query", "")], limit=limit)}, default=str)
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
                return json.dumps({"connected": self._graph.connect(args["from"], args["to"], args["type"], args.get("properties"))}, default=str)
            if action == "forget":
                return json.dumps({"archived": self._graph.forget(args.get("name", ""))})
            if action == "cypher":
                return json.dumps({"rows": self._graph.read_cypher(args.get("query", ""), args.get("properties"))[:limit]}, default=str)
            if action == "stats":
                return json.dumps({"counts": self._graph.stats()})
            if action == "dream":
                return json.dumps({"dream": self._graph.dream()})
            return json.dumps({"error": f"unknown action {action}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def backup_paths(self) -> List[str]:
        return []


def register(ctx) -> None:
    ctx.register_memory_provider(ReverieMemoryProvider())
