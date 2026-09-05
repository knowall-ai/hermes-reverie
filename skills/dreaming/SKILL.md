---
name: dreaming
description: Nightly consolidation of the Reverie graph — review the day, remember what matters, tidy the graph, write a dream diary
version: 0.1.0
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

Gather what happened since the last diary: your Teams chats and channel messages, emails you
sent or received, meeting transcripts and prep notes, work items you touched, and your own
session history (`session_search`). Read for people, organisations, projects, decisions,
risks, and relationships between them.

## 2. Remember, carefully

For every entity worth keeping, use the `reverie` tool:

1. `search` first, always. Never create a second node for someone who exists under another spelling.
2. `remember` with the canonical label (`Person`, `Organization`, `Project`, `Product`, `Concept`,
   `Meeting`, `Decision`, `Risk`) and useful properties: `role`, `company`, `email`, `notes`
   (one or two sentences, dated), `aliases`.
3. `connect` with a typed relationship and a `since`/`date` property:
   works at → `WORKS_AT`; is the X of → `HAS_ROLE {role}`; partners with → `PARTNERS_WITH`;
   introduced by → `INTRODUCED_BY`; interested in → `INTERESTED_IN`; met → `MET_WITH`;
   talked about → `DISCUSSED`; decided → `DECIDED`; owns → `OWNS`; blocked by → `BLOCKED_BY`.
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

## 4. Find connections

Ask the graph a few questions with `cypher` (read-only): who did we meet this month with no
follow-up? Which projects have a `BLOCKED_BY` with no `DECIDED`? Which people appear in two
organisations? Note anything surprising.

## 5. Dream diary

Write `~/.hermes/dreams/YYYY-MM-DD.md`:

- **Remembered:** entities and relationships added or updated (names only)
- **Tidied:** the `dream` report
- **Connections:** two or three things you noticed
- **Tomorrow:** anything that needs a human (a follow-up, a missing owner, a risk)

Keep it under a page. Then stop; you have slept.
