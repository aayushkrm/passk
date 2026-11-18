# AGENTS.md

Guidance for AI coding agents and humans working in this repo.

## What this is

`passk` measures whether a tool-using agent succeeds every time, not just once.
It runs an offline, CI-gated pass^k harness over a deterministic lab-inventory environment
(`labstock`). Tools are served over MCP v2 stdio. Status: scaffold. Most modules are stubs
that raise `NotImplementedError`; see "Layout".

## Setup

```bash
make setup        # uv sync --python 3.11 --extra dev  (no llama-cpp-python)
make lint         # ruff check + ruff format --check
make type         # mypy src
make test         # pytest -m "not model"
```

Local model work needs the optional native build:
`uv pip install -e '.[llama]'` (keep it out of CI; it is compiled on the developer machine).

Other targets (`demo`, `report`, `check-results`, `site`, `release`) are stubs that print a TODO.

## Layout

```
src/passk/schema.py        FROZEN pydantic models and Env/Agent/ToolTransport protocols
src/passk/envs/            base class and labstock environment
src/passk/goals.py         assertion DSL over final state (pure functions)
src/passk/tasks/labstock/  task YAML files (24 planned)
src/passk/transport.py     InProcessTransport, MCPStdioTransport
src/passk/mcp_server.py    MCP v2 server (from mcp.server import MCPServer)
src/passk/agents/          scripted agents, llama-cpp agent (lazy import)
src/passk/runner.py        run(RunConfig) -> RunReport
src/passk/metrics.py       pass@k, pass^k, bootstrap CIs
src/passk/taxonomy.py      primary FailureCategory per trajectory
src/passk/trace.py         OTel GenAI-named spans as JSONL
src/passk/diff.py, gate.py run comparison and CI gate
src/passk/site/            static Pages site and report rendering
src/passk/cli.py           typer app: run, report, diff, gate, serve-tools, validate, demo
scripts/envinfo.py         environment record (copied per repo on purpose)
tests/                     pytest; test_<module>.py next to each owner
configs/                   run and gate configs
results/                   committed, generated run outputs (never hand-edited)
docs/                      design, prereg, deviations, limitations
```

## Invariants

1. `src/passk/schema.py` is frozen. Only the orchestrator changes it. If you need a change,
   write it in your report and keep a local adapter.
2. `results/` is never hand-edited. Results come from `passk run`, and the README and site
   render from them.
3. No network calls in tests. Mark tests that need a GGUF model with `@pytest.mark.model`;
   CI runs `-m "not model"`.
4. Numbers in README and docs are generated between `<!-- results:start -->` and
   `<!-- results:end -->`. Never type a measured number by hand.
5. Demo and scripted-agent output is labelled SYNTHETIC and never shown beside model results
   without that label.
6. Every result directory carries `env.json` from `scripts/envinfo.py`. It records no hostname,
   username or paths.
7. Use the MCP v2 API: `from mcp.server import MCPServer` and `@mcp.tool()`. Do not use
   `FastMCP` (v1 name, removed in v2).
8. Heavy dependencies are optional extras. CI and `passk demo` never need the native build.
9. Do not commit model weights (`*.gguf`) or raw generations. They stay under `models/` or
   `~/.cache/passk`, which are gitignored.
10. Ruff line length is 100. mypy runs with `disallow_untyped_defs` on `src/`.

## Conventions

- Exit codes for the CLI: 0 = pass, 1 = check failed or regression, 2 = usage or runtime error.
- Every CLI command accepts `--json`.
- Commit messages are plain and descriptive. Commits are made in real time (no backdating).
