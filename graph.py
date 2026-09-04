"""Neo4j access for Reverie.

One small class over the official driver. Every query is parameterised; labels and
relationship types are validated against a strict identifier pattern before they are
interpolated, because Cypher cannot parameterise them.

Graph conventions (shared with Sallie's graph so both agents can read one another's data):
labels are capitalised singular (Person, Organization, Project, Product, Concept, Meeting,
Decision); every node has ``name``; matching is case-insensitive on ``name`` and ``aliases``;
relationships are UPPER_SNAKE (WORKS_AT, HAS_ROLE, PARTNERS_WITH, INTRODUCED_BY,
INTERESTED_IN, MET_WITH, DISCUSSED, DECIDED, OWNS, BLOCKED_BY).
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
CANONICAL_LABELS = ("Person", "Organization", "Project", "Product", "Concept", "Meeting", "Decision", "Risk")
WRITE_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|LOAD\s+CSV|CALL\s+\{)\b", re.IGNORECASE)


def _ident(value: str, what: str) -> str:
    if not isinstance(value, str) or not IDENT_RE.match(value):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


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


class Graph:
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None,
                 timeout: float = 3.0):
        from neo4j import GraphDatabase  # lazy: is_available() must not import it

        self._driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout)
        self._database = database or None
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "Graph":
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
            user=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ["NEO4J_PASSWORD"],
            database=os.environ.get("NEO4J_DATABASE") or None,
        )

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    # -- low level ---------------------------------------------------------
    def run(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params, timeout=self._timeout)
            return [record.data() for record in result]

    def ping(self) -> bool:
        try:
            return self.run("RETURN 1 AS ok")[0]["ok"] == 1
        except Exception as exc:
            logger.debug("Reverie ping failed: %s", exc)
            return False

    # -- recall ------------------------------------------------------------
    def recall(self, terms: Iterable[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Nodes whose name/aliases/email contain any term (case-insensitive), with 1-hop neighbours."""
        terms = [t.lower() for t in terms if t]
        if not terms:
            return []
        rows = self.run(
            """
            UNWIND $terms AS term
            MATCH (n)
            WHERE (n.status IS NULL OR n.status <> 'archived')
              AND (toLower(coalesce(n.name, '')) CONTAINS term
                   OR toLower(coalesce(n.email, '')) CONTAINS term
                   OR any(a IN coalesce(n.aliases, []) WHERE toLower(toString(a)) CONTAINS term))
            WITH DISTINCT n
            OPTIONAL MATCH (n)-[r]-(m)
            WITH n, collect({type: type(r), out: startNode(r) = n, name: m.name, label: head(labels(m))})[..8] AS rels
            RETURN elementId(n) AS id, head(labels(n)) AS label, properties(n) AS props, rels
            LIMIT $limit
            """,
            terms=terms, limit=int(limit),
        )
        return rows

    # -- writes ------------------------------------------------------------
    def remember(self, label: str, name: str, props: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """MERGE by case-insensitive name within the label; update properties; stamp timestamps."""
        label = canonical_label(label)
        props = {k: v for k, v in (props or {}).items() if k not in ("name",) and v is not None}
        now = int(time.time())
        rows = self.run(
            f"""
            OPTIONAL MATCH (e:{label}) WHERE toLower(e.name) = toLower($name)
            WITH e LIMIT 1
            CALL {{
              WITH e
              WITH e WHERE e IS NULL
              CREATE (c:{label} {{name: $name, created_at: $now}})
              RETURN c AS node
              UNION
              WITH e
              WITH e WHERE e IS NOT NULL
              RETURN e AS node
            }}
            SET node += $props, node.updated_at = $now
            RETURN elementId(node) AS id, head(labels(node)) AS label, properties(node) AS props
            """,
            name=name.strip(), props=props, now=now,
        )
        if not rows:
            raise RuntimeError("remember matched nothing and created nothing")
        return rows[0]

    def connect(self, from_name: str, to_name: str, rel_type: str,
                props: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rel_type = _ident(rel_type.upper(), "relationship type")
        rows = self.run(
            f"""
            MATCH (a) WHERE toLower(a.name) = toLower($from_name)
            MATCH (b) WHERE toLower(b.name) = toLower($to_name)
            WITH a, b LIMIT 1
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $props, r.updated_at = $now
            RETURN a.name AS from, type(r) AS type, b.name AS to, properties(r) AS props
            """,
            from_name=from_name.strip(), to_name=to_name.strip(), props=props or {}, now=int(time.time()),
        )
        if not rows:
            raise RuntimeError(f"connect: '{from_name}' or '{to_name}' not found — remember them first")
        return rows[0]

    def forget(self, name: str) -> int:
        """Soft delete: archived nodes stop being recalled but keep their history."""
        rows = self.run(
            "MATCH (n) WHERE toLower(n.name) = toLower($name) SET n.status = 'archived', n.archived_at = $now RETURN count(n) AS n",
            name=name.strip(), now=int(time.time()),
        )
        return int(rows[0]["n"]) if rows else 0

    def probe(self, name: str) -> List[Dict[str, Any]]:
        return self.recall([name.strip()], limit=3)

    def read_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if WRITE_RE.search(cypher):
            raise ValueError("cypher action is read-only; use remember/connect/forget for writes")
        return self.run(cypher, **(params or {}))

    # -- dreaming ----------------------------------------------------------
    def dream(self) -> Dict[str, Any]:
        """Deterministic hygiene: merge case-duplicate names within a label, canonicalise
        lowercase labels, count orphans. The LLM-driven part of Dreaming (reading the day's
        conversations and extracting new entities) lives in the skill, not here."""
        report: Dict[str, Any] = {"relabelled": 0, "merged": 0, "orphans": 0}
        for row in self.run("CALL db.labels() YIELD label RETURN label"):
            label = row["label"]
            canon = canonical_label(label) if label.lower() in [c.lower() for c in CANONICAL_LABELS] else None
            if canon and canon != label:
                n = self.run(f"MATCH (n:`{label}`) REMOVE n:`{label}` SET n:{canon} RETURN count(n) AS n")
                report["relabelled"] += int(n[0]["n"]) if n else 0
        for label in CANONICAL_LABELS:
            dupes = self.run(
                f"""
                MATCH (n:{label}) WITH toLower(n.name) AS key, collect(n) AS nodes
                WHERE size(nodes) > 1 RETURN key, [x IN nodes | elementId(x)] AS ids
                """
            )
            for d in dupes:
                keep, *rest = d["ids"]
                for other in rest:
                    self.run(
                        """
                        MATCH (keep) WHERE elementId(keep) = $keep
                        MATCH (dup) WHERE elementId(dup) = $dup
                        CALL apoc.refactor.mergeNodes([keep, dup], {properties: 'discard', mergeRels: true}) YIELD node
                        RETURN node
                        """,
                        keep=keep, dup=other,
                    )
                    report["merged"] += 1
        orphans = self.run("MATCH (n) WHERE NOT (n)--() AND (n.status IS NULL OR n.status <> 'archived') RETURN count(n) AS n")
        report["orphans"] = int(orphans[0]["n"]) if orphans else 0
        return report

    def stats(self) -> Dict[str, int]:
        rows = self.run("MATCH (n) RETURN head(labels(n)) AS label, count(n) AS n ORDER BY n DESC")
        return {r["label"] or "?": int(r["n"]) for r in rows}
