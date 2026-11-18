---
name: passk
description: Use when running, reporting, diffing or gating pass^k reliability evaluations of tool-using agents with the passk CLI. Placeholder until the CLI commands are implemented.
---

# passk

Placeholder skill. It will describe how to:

- run a config with `passk run -c CONFIG` and read `runs/<run_id>/`
- render a report with `passk report RUN_DIR --format md|json|html`
- compare two runs with `passk diff A B`
- gate a candidate with `passk gate A B --policy gate.yaml` (exit 0 pass, 1 regression)

Until these commands are implemented, every command raises `NotImplementedError`.
Do not report results from this skill as measured numbers.
