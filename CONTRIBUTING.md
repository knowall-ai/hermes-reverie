# Contributing to Reverie

Reverie is a generic memory plugin. Anyone can install it against their own graph, their own
agents and their own tools. Keep it that way.

## Rule 1 — nothing in this repo knows who deployed it

The plugin text, the `reverie` tool description and every skill under `skills/` are read by
**every** agent that installs Reverie. They must not name a particular company, person, agent,
chat platform or helper script.

- **Never:** "ask Sallie in Teams", "look them up with `graph.mjs`", "KnowAll's customers",
  "your Azure DevOps work items".
- **Instead:** "ask a colleague or a sibling agent on whatever channel you share", "your
  organisation's directory (Microsoft Graph, Google Workspace, an HR system — whatever your
  skills provide)", "your work-item tracker".
- Examples in guidance use obviously generic names (Acme, Alex) and never a real customer.

Deployment specifics belong in the **agent's own persona** (its `SOUL.md` or equivalent), which
is where an agent learns who its colleagues are and which tools it has. If a piece of guidance
only makes sense for one deployment, it is persona text, not plugin text.

`tests/test_generic.py` fails the build if a deployment name appears in the tool description,
the session prompt block or a skill. Add to its list when you notice a new one slipping in.

## Rule 2 — the graph shape is the product

Guidance you add must reinforce, never dilute, the graph rules in the tool description:

- one node per person, pet, organisation, project, product, meeting or decision;
- one typed edge per relationship, with a `source` property;
- properties only for facts about that node itself — never another entity's name inside a
  `notes`, `family` or `colleagues` property;
- nodes named the way a person would say them (a meeting is "Kick-off call with Acme —
  2026-09-05", not the sentence used to arrange it).

## Pull requests

- Title in conventional-commit form (`feat:`, `fix:`, `docs:`); body with a `Fixes #nnn` line for
  `feat`/`fix`/`perf`/`refactor` and a `## Test plan` section — `pr-lint` enforces this.
- Run `pytest --rootdir=tests tests` locally before pushing.
- Keep secrets, real customer names and real transcripts out of tests and fixtures; use
  low-entropy placeholders.
