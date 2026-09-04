# Reverie

**Graph memory that dreams.** A [Hermes Agent](https://hermes-agent.nousresearch.com) memory
provider by [KnowAll AI](https://knowall.ai): a Neo4j knowledge graph of the people,
organisations, projects, concepts, meetings and decisions an agent meets, recalled before every
turn, plus **Dreaming**, a nightly consolidation that reviews the day and tidies the graph.

```
hermes plugins install knowall-ai/reverie
hermes memory setup reverie
```

## What it does

- **Recall.** Before each non-trivial turn, names, emails and quoted phrases in the prompt are
  matched against the graph (case-insensitive on `name`, `aliases`, `email`) and the hits are
  injected with their relationships:
  `Person Ava Walsh (CEO, Atlantic Pharma) — WORKS_AT Atlantic Pharma; DISCUSSED support renewal`.
- **Remember.** The agent writes through one tool, `reverie`, with `search`, `probe`, `remember`,
  `connect`, `forget`, `cypher` (read-only), `stats` and `dream`. The graph only ever holds curated
  entities, never raw chat.
- **Dreaming.** `hermes memory setup` installs a `dreaming` skill and a 03:00 cron job. Each night
  the agent reviews its conversations, remembers what matters, merges duplicates, fixes labels,
  looks for connections, and writes a dream diary to `~/.hermes/dreams/`.

## Graph conventions

Labels are capitalised singular: `Person`, `Organization`, `Project`, `Product`, `Concept`,
`Meeting`, `Decision`, `Risk`. Every node has `name`; matching is case-insensitive. Relationship
types are `UPPER_SNAKE`: `WORKS_AT`, `HAS_ROLE`, `PARTNERS_WITH`, `INTRODUCED_BY`, `INTERESTED_IN`,
`MET_WITH`, `DISCUSSED`, `DECIDED`, `OWNS`, `BLOCKED_BY`. Archived nodes (`status = 'archived'`)
are kept but never recalled. The same conventions are used by KnowAll's OpenClaw agents, so one
graph can serve several agents.

## Configuration

Secrets in `~/.hermes/.env`:

```
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

Settings in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: reverie
plugins:
  reverie:
    recall_limit: 5
    dreaming_schedule: "0 3 * * *"
```

## Running Neo4j

Any Neo4j 5 with APOC. For a single agent, a Docker container bound to loopback is enough:

```
docker run -d --name neo4j --restart unless-stopped \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -v ~/neo4j/data:/data -e NEO4J_AUTH=neo4j/<password> \
  -e NEO4J_PLUGINS='["apoc"]' neo4j:5-community
```

## Development

```
pip install neo4j pytest
pytest tests
```

`tests/test_terms.py` covers the recall-term extraction; graph tests need a live Neo4j and are
skipped without `NEO4J_PASSWORD`.
