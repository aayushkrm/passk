# AGENTS.md

Guidance for AI coding agents and humans working in this repo.

## What this is

`passk` measures whether a tool-using agent succeeds every time, not just once.
It runs an offline, CI-gated pass^k harness over a deterministic lab-inventory environment
(`labstock`). Tools are served over MCP v2 stdio. The README states the results; `docs/`
records the design, the pre-registration and every deviation from it.

## Setup

```bash
make setup        # uv sync --python 3.11 --extra dev  (no llama-cpp-python)
make lint         # ruff check + ruff format --check
make type         # mypy src
make test         # pytest -m "not model"
```

Local model work needs the optional native build:
`uv pip install -e '.[llama]'` (keep it out of CI; it is compiled on the developer machine).

## Make targets

| Target | What it does |
|---|---|
| `setup` | `uv sync --python 3.11 --extra dev`. |
| `lint` | `ruff check` and `ruff format --check`. |
| `fmt` | `ruff check --fix` and `ruff format`. |
| `type` | `mypy src`. |
| `test` | `pytest -m "not model"`. |
| `demo` | `passk demo`: scripted agents only, output labelled SYNTHETIC. |
| `results` | Writes `results/headline.json` from the committed results. |
| `report` | Runs `results`, then renders the README results block. |
| `check-results` | Fails when the README block or `results/headline.json` is stale. |
| `site` | Builds the static site into `build/site`. |
| `release` | Builds the wheel and sdist into `dist/`. |
| `dogfood` | Validates the smoke set and gates the committed fixtures (run in CI). |

## Layout

```
src/passk/schema.py        pydantic models and Env/Agent/ToolTransport protocols
src/passk/envs/            base class and labstock environment
src/passk/goals.py         assertion DSL over final state (pure functions)
src/passk/tasks/labstock/  24 task YAML files
src/passk/transport.py     InProcessTransport, MCPStdioTransport
src/passk/mcp_server.py    MCP v2 server (from mcp.server import MCPServer)
src/passk/agents/          scripted, llama-cpp and OpenAI-compatible agents (lazy imports)
src/passk/runner.py        run(RunConfig) -> RunReport
src/passk/metrics.py       pass@k, pass^k, bootstrap CIs
src/passk/taxonomy.py      primary FailureCategory per trajectory
src/passk/trace.py         OTel GenAI-named spans as JSONL
src/passk/validate.py      estimator and gate calibration (E1, E2)
src/passk/diff.py, gate.py run comparison and CI gate
src/passk/envinfo.py       environment record (scripts/envinfo.py is a thin wrapper)
src/passk/numfmt.py        shared number formatting for reports, gate output and the README
src/passk/site/            static Pages site and report rendering
src/passk/cli.py           typer app: run, report, diff, gate, serve-tools, validate, site, demo
scripts/render_readme.py   README results block and results/headline.json
tests/                     pytest; test_<module>.py next to each owner
configs/                   run and gate configs
results/                   committed, generated run outputs (never hand-edited)
docs/                      design, prereg, deviations, limitations, failure analysis
```

## Invariants

1. Change `src/passk/schema.py` only together with a `schema_version` decision and an entry
   in `docs/deviations.md`.
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
- Commit messages are plain and descriptive. Do not backdate commits.
