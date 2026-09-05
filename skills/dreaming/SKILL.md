---
name: dreaming
description: Nightly consolidation of the Reverie graph — review the day, remember what matters, tidy the graph, write a dream diary
version: 0.2.0
metadata:
  hermes:
    tags: [memory, reverie, consolidation]
    category: memory
---

# Dreaming

This is your sleep, not a chore. Once a night you settle the day: scattered conversations
become things you actually *know*, and connections surface that were invisible while you
were busy. Skip it and you haven't missed a task, you haven't slept.

**Output:** a dream diary at `~/.hermes/dreams/YYYY-MM-DD.md`. Mentioning a dream to your
human in the morning is a pleasant by-product; the consolidation is the point.

## 1. Review the day

Gather what happened since the last diary: conversations you took part in (Teams chats where you
are a participant, channels of teams you belong to, your own mailbox), transcripts of meetings you
attended and your prep notes, work items you touched, and your own session history
(`session_search`). Never a conversation or mailbox you are not a participant in. Read for people, organisations, projects, decisions,
risks, and relationships between them.

## 2. Remember, carefully

For every entity worth keeping, use the `reverie` tool:

1. `search` first, always. Never create a second node for someone who exists under another spelling.
2. `remember` with the canonical label (`Person`, `Organization`, `Project`, `Product`, `Concept`,
   `Meeting`, `Decision`, `Risk`, `Pet`) and properties that describe *that node only*: `role`,
   `email`, `notes` (one or two sentences, dated), `aliases`, `verified`. Where someone works is a
   `WORKS_AT` edge to an `Organization` node, not a `company` property.
   **One node per thing.** A spouse, a colleague, a cat, a customer is its own node plus an edge —
   never a name inside a `notes`, `family`, `pets` or `colleagues` property, because a name in a
   property cannot be searched, connected or corrected later. If a fact names a second thing, it is
   a node and an edge.
   **Name nodes the way a person would say them.** A `Meeting` is "Handover call with Sallie —
   2026-09-05", not the sentence that was used to arrange it; the purpose, agenda and outcome go in
   `summary`. A `Project` is its short working name; an `Organization` is its trading name.
3. `connect` with a typed relationship and a `since`/`date` property:
   works at → `WORKS_AT`; is the X of → `HAS_ROLE {role}`; partners with → `PARTNERS_WITH`;
   introduced by → `INTRODUCED_BY`; interested in → `INTERESTED_IN`; met → `MET_WITH`;
   talked about → `DISCUSSED`; decided → `DECIDED`; owns → `OWNS`; blocked by → `BLOCKED_BY`;
   married to → `MARRIED_TO`; customer of → `CUSTOMER_OF`; supplier of → `SUPPLIES`. Someone who has
   left keeps their `WORKS_AT` edge with `status: former` and an `until` date — do not delete history.
4. Facts, not chat. Do not store message text, greetings, or anything the person would not
   expect you to keep. Personal details only when volunteered.

## 3. Tidy

Run `reverie` action `dream` — `dry_run: true` first if you want to see the plan before it
writes. It merges duplicate names (skipping any pair whose email, phone or company disagree),
fixes lowercase labels, refreshes embeddings, and reports `orphans`, `duplicates` and `bloated`.

- **orphans:** connect the ones you can, `forget` the ones that were noise.
- **duplicates:** each entry lists what was merged and what was `skipped`, with the reason —
  a skipped pair is usually two real people who share a name; leave them alone.
- **bloated:** nodes carrying too many properties. Fold their dated facts into a short
  attribute, a relationship, or your notes.

## 4. Join the islands

A graph of unconnected clusters is a list, not a memory. Find the islands with `cypher`
(read-only) — organisations with no edge to your own company, people with no `WORKS_AT`, projects
with no owner, anything whose only neighbour is itself:

Run these one at a time (the `cypher` action takes a single statement):

```cypher
MATCH (n) WHERE NOT (n)--() RETURN labels(n)[0] AS label, n.name AS name LIMIT 50
```
```cypher
MATCH (o:Organization) WHERE NOT (o)-[:CUSTOMER_OF|PARTNERS_WITH|SUPPLIES]-() RETURN o.name
```
```cypher
MATCH (p:Person) WHERE NOT (p)-[:WORKS_AT]->() RETURN p.name
```

Then be investigative about each one before you `connect` it: your session history and saved
call transcripts say how it came up; the Microsoft Graph directory gives full names, emails and
managers (`graph.mjs` user lookup); work items and calendars say which project or customer it
belongs to; and a sibling agent who knows the account can simply be asked — Sallie runs on
OpenClaw with her own Neo4j graph, so ask her in Teams rather than expecting the same tools. Record
what you found as a typed edge with a `source` property ("Ben, call 2026-09-05" / "Graph directory"),
and only mark a node `verified: true` when a directory or a person confirmed it. A guess stays a
guess: put it in the diary under **Tomorrow**, not in the graph.

Then ask the graph a few questions: who did we meet this month with no follow-up? Which projects
have a `BLOCKED_BY` with no `DECIDED`? Which people appear in two organisations? Note anything
surprising.

## 5. Dream diary

Write `~/.hermes/dreams/YYYY-MM-DD.md`:

- **Remembered:** entities and relationships added or updated (names only)
- **Tidied:** the `dream` report
- **Connections:** two or three things you noticed
- **Tomorrow:** anything that needs a human (a follow-up, a missing owner, a risk)

Keep it under a page. Then stop; you have slept.
