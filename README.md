# Reverie

**Graph memory that dreams.** A [Hermes Agent](https://hermes-agent.nousresearch.com) memory
provider by [KnowAll AI](https://knowall.ai): a Neo4j knowledge graph of the people,
organisations, projects, concepts, meetings and decisions an agent meets, recalled before every
turn, plus **Dreaming**, a nightly consolidation that reviews the day and tidies the graph.

```
npm install -g github:knowall-ai/mcp-reverie#feat/semantic-search   # the graph server
hermes plugins install knowall-ai/reverie
hermes memory setup reverie
```

## What it does

- **Recall.** Before each non-trivial turn, names, emails and quoted phrases in the prompt are
  matched against the graph — hybrid keyword **and** semantic search, so "Ben Weeks" also finds
  "Benjamin Weeks" — and the hits are injected with their relationships:
  `Person Ava Walsh (CEO, Atlantic Pharma) — WORKS_AT Atlantic Pharma; DISCUSSED support renewal`.
- **Remember.** The agent writes through one tool, `reverie`, with `search`, `probe`, `remember`,
  `connect`, `forget`, `cypher` (read-only), `stats` and `dream`. The graph only ever holds curated
  entities, never raw chat.
- **Dreaming.** `hermes memory setup` installs a `dreaming` skill and a 03:00 cron job. Each night
  the agent reviews its conversations, remembers what matters, merges duplicates, fixes labels,
  looks for connections, and writes a dream diary to `~/.hermes/dreams/`.

## How it talks to the graph

This plugin is a **thin client**. It never opens a Bolt connection; the graph lives behind
[mcp-reverie](https://github.com/knowall-ai/mcp-reverie), an MCP server (binary `reverie`, alias
`mcp-neo4j-agent-memory`), and the plugin spawns it as a child process and speaks JSON-RPC 2.0
over its stdin/stdout:

```
Hermes ──▶ ReverieMemoryProvider (__init__.py)
             └─ graph.py        verbs → tools     (recall, remember, connect, forget, dream…)
                 └─ mcp_client.py  stdio MCP      (initialize → tools/call, restart on crash)
                     └─ reverie (mcp-reverie)     ── Bolt ──▶ Neo4j
```

| Reverie verb | MCP tool |
| --- | --- |
| `search` / recall, `probe` | `search_memories` (hybrid by default; `search_mode`, `similarity_threshold`, `depth`, `limit`) |
| `remember` | `search_memories` (exact-name dedupe) then `create_memory` or `update_memory` |
| `connect` | `search_memories` ×2 to resolve names (`from_label`/`to_label` when a name is ambiguous), then `create_connection` |
| `forget` | `update_memory` (`status = 'archived'`) for a `name`, `delete_memory` with `hard`, or `delete_connection` given `from` + `to` + `type` together |
| `cypher` | `query_memories` (read-only, 200 rows, 10 s) |
| `stats` | `memory_stats` |
| `dream` | `dream` (`dry_run` supported) |

One process serves the whole session: requests are serialised and demultiplexed by JSON-RPC id,
so several turns can call at once, and a server that dies is respawned. A call is retried only
when that cannot repeat it: when the request provably never left the client, or when the process
has exited and the call was read-only. **A timeout while the server is still running is never
retried** — the first call after a cold start pays for the embedding model, and a slow server is
still a working one, so the error is reported and the late answer discarded as a stale id.
The plugin validates labels, relationship types, node ids, properties and search arguments
against the server's contract (mcp-reverie `feat/semantic-search` @ `01dceae`) before calling, so
a bad call comes back as a readable error instead of a rejected tool call.

**Names are resolved, not guessed.** `connect` and `forget` look each end up by exact name — a
full page of candidates, filtered by label where one is given. If a name matches more than one
memory ("Atlas" the Project and "Atlas" the Person) the call is refused with both labels named,
and `from_label` / `to_label` settle it. `remember` keeps taking the first match within its own
label, since two of those are a duplicate for Dreaming to merge.

**`forget` still archives by default.** The server's `delete_memory` is a hard delete, so `forget`
maps to `update_memory` setting `status = 'archived'` (archived nodes are dropped from recall,
exactly as before). Pass `hard: true` to delete outright.

## Graph conventions

Labels are capitalised singular: `Person`, `Organization`, `Project`, `Product`, `Concept`,
`Meeting`, `Decision`, `Risk` — and they are Cypher identifiers, so free text such as
"Atlantic Pharma" is rejected as a label (it is a `name`, not a label). Every node has `name`;
matching is case-insensitive. Relationship types are `UPPER_SNAKE`: `WORKS_AT`, `HAS_ROLE`,
`PARTNERS_WITH`, `INTRODUCED_BY`, `INTERESTED_IN`, `MET_WITH`, `DISCUSSED`, `DECIDED`, `OWNS`,
`BLOCKED_BY`. Archived nodes (`status = 'archived'`) are kept but never recalled. Property values
are text, numbers, booleans or lists of those — anything richer belongs in its own memory. The
same conventions are used by KnowAll's other Reverie clients, so one graph can serve several
agents.

## Installing the server

Until `@knowall-ai/reverie` is published, install it from GitHub:

```
npm install -g github:knowall-ai/mcp-reverie#feat/semantic-search   # today
npm install -g github:knowall-ai/mcp-reverie                        # once the PR lands on main
npm install -g @knowall-ai/reverie                                  # once published to npm
which reverie   # must be on PATH for the plugin to report itself available
```

Requires Node 18+ and **Neo4j 5.9 or newer** (`dream` uses `COUNT {}` and `IS :: STRING`), with
APOC for duplicate merging.

## Configuration

Secrets in `~/.hermes/.env` — the plugin passes every `NEO4J_*` and `REVERIE_*` variable through
to the server process and never logs their values:

```
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
REVERIE_EMBEDDINGS=local          # local | openai | azure | ollama | voyage | none
REVERIE_MODEL_CACHE=/var/lib/hermes/reverie-models   # optional, for the local model
```

Settings in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: reverie
plugins:
  reverie:
    server_command: reverie       # how to start the MCP server
    embeddings: local             # local | openai | azure | ollama | voyage | none
    search_mode: hybrid           # hybrid | keyword | semantic
    similarity_threshold: ""      # blank = the server's default (0.4)
    recall_limit: 5
    recall_depth: 1               # relationship hops recalled with each entity, 0-5
    server_timeout: 30            # seconds per tool call
    server_startup_timeout: 60    # the local embedding model downloads on first use
    dreaming_schedule: "0 3 * * *"
```

Other server-side knobs, set in `.env`: `REVERIE_EMBEDDING_MODEL`, `REVERIE_EMBED_TIMEOUT_MS`
(default 30000, max 120000), `REVERIE_LAZY_EMBED_BATCH` (default 100), `REVERIE_MAX_PROPERTIES`
(default 30 — above this, `dream` reports a node as bloated and `update_memory` returns a hint).

## Running Neo4j

Any Neo4j 5.9+ with APOC. For a single agent, a Docker container bound to loopback is enough:

```
docker run -d --name neo4j --restart unless-stopped \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -v ~/neo4j/data:/data -e NEO4J_AUTH=neo4j/<password> \
  -e NEO4J_PLUGINS='["apoc"]' neo4j:5-community
```

## Development

```
pip install pytest
pytest --rootdir=tests tests   # --rootdir: the repo root is itself a package, and a hyphenated folder name breaks collection
```

No runtime dependencies and nothing to install beyond pytest: `tests/fake_mcp_server.py` is a
stand-in MCP server that speaks the same stdio JSON-RPC and enforces the same argument contract,
so the client (handshake, round trip, tool errors, timeouts, crash-restart, thread safety) and
every `graph.py` mapping are covered without Neo4j or the real server. `tests/test_terms.py` also
holds a live round trip that runs only when `reverie` is on PATH and `NEO4J_PASSWORD` is set.
