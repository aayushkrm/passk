@AGENTS.md

## Claude Code notes

- The usage skill for this repo is `.claude/skills/passk/SKILL.md`. Read it before running the CLI.
- Change `src/passk/schema.py` only together with a `schema_version` decision and an entry in `docs/deviations.md`.
- Use `make lint type test` before reporting a task as done.
- Write any sample or demo output as SYNTHETIC unless it comes from a model run.
