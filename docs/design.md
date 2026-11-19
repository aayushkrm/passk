# Design

This file records the decisions behind `passk` and why they were made. The interfaces
are in `src/passk/schema.py` and are not described again here.

## What is measured

- **pass^k.** The probability that an agent succeeds on all k independent attempts at a task.
  The estimator is the unbiased one over n trials: `C(c, k) / C(n, k)` per task, averaged over
  tasks (`passk.metrics.pass_hat_k`). The same trials give pass@k, `1 - C(n - c, k) / C(n, k)`.
- **Success.** A deterministic check of the final environment state against the task's goal
  assertions, with no policy violation and no forbidden assertion. There is no LLM judge.
- **Failure taxonomy.** One primary category per episode, in a fixed precedence
  (`passk.taxonomy`): policy violation, schema error, unrecovered tool error, false success
  claim, loop, max turns, wrong state. A recovered tool error inside a passing episode is not a
  failure.

## Environment and tasks

- `labstock` is a deterministic lab-inventory environment with a budget, orders and an approval
  policy for controlled compounds. The state can be read after an episode, so success needs no
  judge.
- 24 tasks in three tiers: single call, lookup then act, and policy-constrained or with
  near-duplicate names. Each task carries a reference solution. The reference solution is what
  the oracle agents replay and what the calibration uses for reference lengths.
- Optional transient failures (`flaky_p`) are set through `env_params` in a `RunConfig`. The
  in-process environment and the MCP server both receive the value. The recovery experiment E6
  uses it (`configs/e6_flaky.yaml`; see `docs/deviations.md`, D16).

## Transports

- `InProcessTransport` calls the environment directly. It is the default for scripted runs and
  calibration, because it is fast.
- `MCPStdioTransport` runs `python -m passk.mcp_server labstock` as a subprocess and speaks MCP
  v2 over stdio. Every model run uses it.
- The two transports give identical trajectories for the oracle on all 24 tasks, and for a noisy
  oracle including its policy violations. `tests/test_transport_integration.py` checks this on
  every test run.
- MCP v2 (`from mcp.server import MCPServer`) follows the 2026-07-28 specification, which has no
  sessions. Per-episode state lives in the server under an `env_handle` minted by
  `episode_reset` and passed as an explicit argument on every call. The control tools are
  `episode_reset`, `episode_state` and `episode_violations`.

## Runner

- `runner.run(config)` runs tasks in id order, then trials. Each episode's seed is a SHA-256 of
  (base seed, task id, trial), so results do not depend on the process or on the hash seed.
- The tool list is fetched once per run. Agents receive it on every step.
- A run directory is written only after all episodes finish, through atomic renames. A run
  whose directory already holds files is refused, so a result is never overwritten.
- `light=True` and `write=False` give an in-memory run without spans, the environment record or
  the summary bootstrap. Calibration uses this path. Its per-task statistics are identical to a
  written run with the same seeds.

## Estimator validation (E1) and gate calibration (E2)

- E1 uses NoisyOracle agents, whose success probability per task is known in closed form:
  `(1 - p) ** m_t`, where m_t is the reference length. The check is therefore against an
  analytic value, not against a model. The z-score uses the exact standard error of the
  unbiased estimator under that binomial model, so a deviation is judged against its own
  sampling noise.
- E2 runs seeded A/A and A/B comparisons through the real gate. Its trial count, 5 per task,
  equals the main experiment's, so the gate is calibrated at the operating point it is used at.
- Both calibrations are labelled SYNTHETIC. They test the tool, not a model.

## Gate

- `gate(baseline, candidate, policy)` compares mean pass^k at k=K over the shared tasks. The
  delta is the candidate minus the baseline, with a paired bootstrap over tasks (10,000
  resamples, seed 0).
- The gate fails when the drop exceeds `max_drop` and the bootstrap interval for the delta
  excludes 0 (when significance is required), or when the candidate falls below an absolute
  floor. A drop larger than `max_drop` with an interval that includes 0 is reported as "not a
  regression".

## Scripted agents and labels

- Scripted agents (`oracle`, `noisy_oracle`, `replay`) set `env_info.synthetic = true`. Every
  report built from them carries the SYNTHETIC label into the CLI output, the gate output, the
  site and the README.

## Local model agent

- `LlamaCppAgent` calls `create_chat_completion` with a JSON-schema response format. The
  schema is a flat object with four required keys: `kind`, `tool` (an enum of the environment's
  tools plus `"none"` for a final action), `arguments` (an object, `{}` for a final action) and
  `message` (empty for a tool call, the reply for a final action). A flat schema keeps grammar
  compilation cheap. Before this change only `kind` was required, and a model could omit the
  tool name; see `docs/deviations.md`.
- Output that does not parse becomes a schema-error action, and the episode continues.
- The per-call seed is the episode seed plus the number of past steps. The model is loaded with
  `n_gpu_layers=0`. Each run records the model's repo id, file name and SHA-256 in `env.json`.
- `llama_cpp` is an optional extra. It is imported only when the agent is built, so CI and
  `passk demo` never need it.

## Hosted arm

- `OpenAICompatAgent` calls a chat endpoint (OpenRouter by default) with the same action
  prompt. It paces requests, retries rate limits and server errors with backoff, and records
  the provider-reported model id per response. The API key is read from an environment variable
  and is never stored in a config, a result or a log.
- A provider error aborts the run, and nothing is written. An outage is not counted as a model
  failure.

## Reproducibility

- The environment record (`scripts/envinfo.py`) holds the git SHA and dirty flag, UTC time,
  CPU model and selected flags, core counts, RAM, OS, Python and package versions. It holds no
  hostname, username or path.
- Results are rendered from JSON. `scripts/render_readme.py` writes the README results block and
  `results/headline.json`. `make check-results` fails when either is stale.

## Failure taxonomy rules

- The checking order is: any recorded violation gives `policy_violation`; otherwise a met goal
  gives `success`; otherwise the first of `schema_error`, `tool_error_unrecovered`,
  `false_success_claim`, `loop`, `max_turns` and `wrong_state` that holds.
- `success` is checked before the failure rules. An episode that meets its goal with no
  violation passes, even if it recovered from errors on the way.
- The schema-error rule and the tool-error rule differ on purpose. A schema error is a model
  output that is not a valid action under the contract. The harness counts it whenever it
  happens, even if the model later repairs it, because that output would not have been accepted
  by a tool client. A tool error is an answer from the environment. The agent is expected to
  handle those, so an error the agent recovered from is not a failure.

## Failure counts

- `TaskStats.failures` and `RunReport.failure_totals` count unsuccessful trials only. A pass
  is not a failure category.
- The runner computes these through `metrics.aggregate_task`, and the summary through
  `metrics.summarize_tasks`. The unit tests check the same functions, and a runner test compares
  a written report with them.

## Headline contract (`results/headline.json`)

- The file follows the hub's standard contract (`schemas/headline.schema.json`, schema version
  1): `tool`, `tagline`, `quickstart`, `environment`, at most six `items`, and `links`. It is
  generated by `scripts/render_readme.py` from the committed results JSON only. Nothing is typed by
  hand.
- An item is present only when its source result exists, and its `source` is the results JSON it
  came from. Items are chosen in this priority order, and the first six are kept: the E3 gap for
  Qwen2.5-0.5B and for Qwen2.5-1.5B (both featured), the E5 gate verdict, the E7 run (partial if
  not complete), the E1 estimator error, the E2 gate calibration, and the optional Qwen3-1.7B gap.
- The E3 gap is the mean per-task pass@1 minus pass^5. Its interval is a seeded percentile bootstrap
  over tasks (`metrics.bootstrap_ci`, 10,000 resamples). The failure-category leader is in the
  E3 item's `detail`, as the category with the largest share of episodes. Success is not counted.
- The E5 verdict is `passk.gate.gate` applied to the baseline and candidate reports with
  `configs/gate.yaml`. It is not read from a text file. A partial run gives "not evaluated", never a
  verdict.
- The E1 and E2 items are labelled synthetic. A partial E3 or E7 run has "PARTIAL" in its value.
- The README results block lists the same item strings, and `make check-results` fails when either
  output is stale.
