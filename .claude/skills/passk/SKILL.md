---
name: passk
description: Use when running, reporting, diffing or gating pass^k reliability evaluations of tool-using agents with the passk CLI, or when checking whether a passk report or gate result is trustworthy.
---

# passk

`passk` runs tasks x trials for a tool-using agent, reports pass@k and pass^k with
intervals, compares two runs, and gates a candidate against a baseline. Every command
accepts `--json`.

## Exit codes

- `0`: pass, or the command completed.
- `1`: a check failed (`passk gate` found a regression or a floor breach).
- `2`: usage or runtime error. Errors print `passk: error: ...` to stderr. With `--json`,
  a JSON object `{"error": ..., "exit_code": 2}` also goes to stdout.

## Commands

| Command | What it does |
|---|---|
| `passk run -c CONFIG` | Runs a RunConfig YAML and writes `runs/<run_id>/` (report.json, trajectories.jsonl, spans.jsonl, env.json). |
| `passk report RUN_DIR --format md\|json\|html [--out FILE]` | Renders one run. `--json` prints the report JSON whatever `--format` says. |
| `passk diff A B [--k K]` | Per-task pass^k deltas, worst regression first, plus mean pass^k per k and the failure-category shift. Read-only; exit 0. |
| `passk gate A B --policy gate.yaml` | Exit 0 if the candidate passes, 1 if it regresses. |
| `passk serve-tools labstock` | Serves the labstock tools over MCP v2 stdio. Stdout is the protocol stream. |
| `passk validate [--k 3] [--max-drop 0.10] [--pairs 200]` | Calibration of the estimator (E1) and the gate (E2) on scripted agents. Output is SYNTHETIC. |
| `passk demo` | Scripted-agent demo, under 30 s, no model. Output is SYNTHETIC. The gate verdict it prints is FAIL by design, and the exit code is 0 whatever the verdict. Do not use the demo's exit code as a gate result; use `passk gate` on two run directories for that. |

`A` and `B` are run directories (or a `report.json` path). `diff` and `report` also read
`trajectories.jsonl` when it is present. With it, the failure shift is a paired
baseline-by-candidate matrix over `(task_id, trial)`. Without it, only totals are shown.

## Gate policy file

Top-level keys only. Unknown keys are rejected, so typos fail with exit 2.

```yaml
k: 3                      # pass^k is compared at this k (required, >= 1)
max_drop: 0.10            # largest tolerated drop in pass^k (required, >= 0)
min_pass_hat_k: 0.50      # absolute floor on the candidate's mean pass^k (optional)
require_significance: true  # a drop also needs a CI that excludes 0 (default true)
alpha: 0.05               # CI is 1 - alpha wide (default 0.05)
```

Gate logic, with `delta = candidate - baseline` in mean pass^k over tasks:

- Regression: `-delta > max_drop` AND (when `require_significance`) the paired bootstrap
  CI for delta lies below 0. The bootstrap resamples tasks, 10,000 times, seed 0.
- Floor: candidate mean pass^k below `min_pass_hat_k` fails regardless of significance.
- Both runs must have the same task ids. Otherwise exit 2.

The reasons list states each number and why the decision was made. Quote the reasons
rather than only the PASS/FAIL line.

## Reading the output honestly

- A drop above `max_drop` that is not significant passes. The reason says so. Report it as
  "not distinguishable at this n", not as "no change".
- Pass^k with n trials per task is noisy at small n. Check the trial count before trusting
  a gate with few tasks.
- A report or site page marked SYNTHETIC came from scripted agents or a fixture. Never put
  it beside model results without that label. The `env_info.synthetic` field decides it.
- Every number comes from the report files. Do not type measured numbers into docs.

## Site

`passk report RUN_DIR --format html --out page.html` writes one standalone page. The build
function `passk.site.build.build_site(results_dir, out_dir)` writes `index.html`, `llms.txt`
and `runs/<run_id>/{index.html,report.json}`. Pages load nothing from the network.

## Schema

`schemas/run_report.schema.json` is generated from `RunReport.model_json_schema()`. A test
fails if the committed file differs from the generated one.

## Fixtures

`tests/fixtures/report_*.json` are hand-chosen SYNTHETIC RunReports for tests. They are not
measurements. Their `env_info.synthetic` is true.
