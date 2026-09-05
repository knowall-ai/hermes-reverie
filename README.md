# Reverie

![Reverie — graph memory that dreams](./images/reverie-banner.png)

**Graph memory that dreams.** A [Hermes Agent](https://hermes-agent.nousresearch.com) memory
provider by [KnowAll AI](https://knowall.ai): a Neo4j knowledge graph of the people,
organisations, projects, concepts, meetings and decisions an agent meets, recalled before every
turn, plus **Dreaming**, a nightly consolidation that reviews the day and tidies the graph.

```
npm install -g github:knowall-ai/mcp-reverie   # the graph server, 0.4.x or newer
hermes plugins install knowall-ai/hermes-reverie
hermes memory setup reverie
```

## Why Reverie

Most agent memory is a pile of facts: a text file, a vector store, or a list of triples. Reverie is
a **map of the entities in an agent's world and how they relate**, and it keeps that map healthy.

- **A typed entity graph, not a fact store.** `Person`, `Organization`, `Project`, `Place`,
  `Concept`, `Meeting`, `Decision` and the rest are first-class nodes with typed relationships
  (`WORKS_AT`, `DISCUSSED`, `DECIDED`, `SPONSORS`…). Ask "Who at Atlantic Pharma have we talked to
  about the support renewal?" and the answer is a graph walk, not a similarity search.
- **Recall happens automatically, every turn.** The provider extracts names, emails and quoted
  phrases from the prompt and injects the matching entities with their relationships before the
  model answers. Nothing depends on the model remembering to call a tool.
- **It dreams.** Each night the agent reviews the day's conversations, remembers what matters,
  merges duplicates, fixes labels, re-embeds, looks for missing connections and writes a dream
  diary. Bloated nodes (facts piled on as properties) are flagged so they get modelled properly.
- **One graph, any agent.** The graph is served by [mcp-reverie](https://github.com/knowall-ai/mcp-reverie)
  over the Model Context Protocol, so agents on other frameworks (OpenClaw, Claude Desktop, Azure
  AI Foundry) share memory with Hermes agents. KnowAll runs an OpenClaw agent and a Hermes agent
  against one graph this way; this plugin is the Hermes-side client of that server.
- **Search that finds "Ben" when you say "Benjamin".** Hybrid keyword + semantic search with
  local embeddings by default (no API key), switchable to OpenAI, Azure OpenAI, Ollama or Voyage.
- **You can see it think.** A brain view API (in mcp-reverie) streams recalls, writes and dreams
  live, so the [Agents Portal](https://github.com/knowall-ai/agents-portal) can draw the graph
  lighting up.
- **Yours to run.** Neo4j on your own box, no hosted service required, no data leaves the VM
  unless you choose a remote embedding provider.

Compared with the memory providers bundled with Hermes (Mem0, Honcho, Hindsight, Supermemory,
OpenViking and others): those are mature and several offer hosted tiers, which Reverie does not.
Hindsight and Mnemosyne also build graphs, but as an entity layer over facts or a generic triple
store. None run a scheduled consolidation with hygiene reporting, share one graph across agent
frameworks, or expose a live graph visualisation. Reverie is the right choice when the agent's
value lies in *who* and *what* it knows and how those connect; it is not the right choice if you
want a managed cloud memory with a track record.

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
| `remember` | `search_memories` with `search_mode: "exact"` (dedupe) then `create_memory` or `update_memory` |
| `connect` | `search_memories` ×2 (`search_mode: "exact"`) to resolve names (`from_label`/`to_label` when a name is ambiguous), then `create_connection` |
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
against the server's contract before calling, so a bad call comes back as a readable error
instead of a rejected tool call.

**Minimum server: mcp-reverie `95baf2a` (0.4.x) or newer.** The plugin is a thin client and
leans on two things that landed in [mcp-reverie #20](https://github.com/knowall-ai/mcp-reverie/pull/20):
`search_memories` accepts `search_mode: "exact"` (case-insensitive equality on `name`, an alias
or `email`), and archived memories are excluded from results *and* from the connections
returned with them unless `include_archived: true` is passed. Against an older server, name
lookups fail with an invalid-argument error and archived nodes reappear in recall.

**Names are resolved, not guessed.** `connect` and `forget` look each end up with the server's
exact mode: only a memory whose `name`, alias or `email` equals what was typed, narrowed by
label where one is given, so a common first name cannot crowd the real node out of the page and
an alias still finds the node it belongs to. If a name matches more than one memory ("Atlas" the
Project and "Atlas" the Person) the call is refused with both labels named, and `from_label` /
`to_label` settle it. `remember` keeps taking the first match within its own label, since two of
those are a duplicate for Dreaming to merge.

**`forget` still archives by default.** The server's `delete_memory` is a hard delete, so `forget`
maps to `update_memory` setting `status = 'archived'`. The server then hides the node: it is gone
from recall, from name lookups, and from the neighbourhoods of the nodes that are still returned.
Nothing in the plugin asks for archived memories back — `graph.search_memories(...,
include_archived=True)` is there for a caller that needs to. Pass `hard: true` to delete outright.

## Graph conventions

Labels are capitalised singular: `Person`, `Organization`, `Project`, `Product`, `Concept`,
`Meeting`, `Decision`, `Risk` — and they are Cypher identifiers, so free text such as
"Atlantic Pharma" is rejected as a label (it is a `name`, not a label). Every node has `name`;
matching is case-insensitive. Relationship types are `UPPER_SNAKE`: `WORKS_AT`, `HAS_ROLE`,
`PARTNERS_WITH`, `INTRODUCED_BY`, `INTERESTED_IN`, `MET_WITH`, `DISCUSSED`, `DECIDED`, `OWNS`,
`BLOCKED_BY`. Archived nodes (`status = 'archived'`) are kept but never recalled — the server
leaves them out unless a call asks for `include_archived`. Property values
are text, numbers, booleans or lists of those — anything richer belongs in its own memory. The
same conventions are used by KnowAll's other Reverie clients, so one graph can serve several
agents. A property is for a durable attribute (email, role, phone); anything dated or episodic
belongs in a relationship, its own node, or the agent's notes — the graph is a map of entities,
not a notebook.

## Installing the server

Until `@knowall-ai/reverie` is published, install it from GitHub:

```
npm install -g github:knowall-ai/mcp-reverie   # today
npm install -g @knowall-ai/reverie            # once published to npm
which reverie   # must be on PATH for the plugin to report itself available
```

Requires Node 18+ and **Neo4j 5.9 or newer** (`dream` uses `COUNT {}` and `IS :: STRING`), with
APOC for duplicate merging. The server must be **mcp-reverie `95baf2a` (0.4.x) or newer**: the
plugin's name lookups use `search_mode: "exact"` and rely on the server hiding archived
memories. An older server rejects the search argument and shows archived nodes in recall.

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
    search_mode: hybrid           # hybrid | keyword | semantic (the recall default)
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

## Licence

MIT, © KnowAll AI Ltd. See [LICENSE](./LICENSE). Commercial support and hosted options:
hello@knowall.ai.
