<!--
Title format: conventional-commit prefix + short description.
  feat: ...      fix: ...      perf: ...      refactor: ...     (require Fixes #nnn)
  chore: ...     ci: ...       docs: ...      deps: ...
  test: ...      build: ...    style: ...
-->

## Summary

<!-- 1–3 bullets: what changed and why. -->

## Fixes

Fixes #

<!-- At least one `Fixes #nnn` line is required for feat/fix/perf/refactor (the lint checks for one; list every issue you resolve). Optional for every other prefix (chore, ci, docs, deps, test, build, style) — delete the section if unused. -->

## Test plan

<!--
Steps a reviewer can follow, with expected results, e.g.
  1. `pip install neo4j pytest && pytest --rootdir=tests tests`
  2. With `NEO4J_PASSWORD` set against a live Neo4j 5, the graph tests run too.
  3. **Expected:** all tests pass.
For refactor / docs / CI / deps PRs: N/A — <reason>
-->
