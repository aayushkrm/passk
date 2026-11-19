# passk

pass^k for tool-using agents: an offline harness that measures whether an agent succeeds on every attempt, not just on one.

> **Status.** The harness, the 24-task LabStock environment, the MCP server, the calibration (E1 and E2), the demo, and the E3, E4 and E5 model runs are built, run and committed. The E5 gate verdict is a pass in the pre-registered direction (see Findings). The E6 robustness run, the Qwen3-1.7B contrast arm and the E7b descriptive hosted run are also run and committed. The full hosted design (E7) has not been run: the free-tier daily quota does not allow it. Everything under "Results" is rendered from committed JSON in `results/` by `make report`.

## Quickstart

```bash
uv tool install git+https://github.com/aayushkrm/passk
passk demo
```

`passk demo` runs scripted agents only, needs no model and no download, and finishes in well under 30 seconds. Its output is labelled **SYNTHETIC**. The repository is not published yet, so until then run it from a clone: `make setup && uv run passk demo`.

## What it measures

For one task, run n independent trials and count c successes.

- **pass@k** is the chance that at least one of k attempts succeeds: `pass@k = 1 - C(n - c, k) / C(n, k)`.
- **pass^k** is the chance that all k attempts succeed: `pass^k = C(c, k) / C(n, k)`. This is the unbiased estimator. The reported value is the mean over tasks.

The gap between pass@1 and pass^5 shows how much an agent's success depends on luck. A success counts only when the final environment state meets every goal assertion, with no policy violation and no forbidden assertion. There is no LLM judge.

Each episode gets one primary category, checked in this order: `policy_violation` (any recorded violation), then `success` (the goal is met with no violation), then `schema_error`, `tool_error_unrecovered`, `false_success_claim`, `loop`, `max_turns` and `wrong_state`. A passing episode is never a failure. See [`docs/design.md`](docs/design.md).

## The environment

`labstock` is a deterministic lab-inventory environment. Compounds have a stock, a hazard class (`none`, `flammable`, `toxic`, `controlled`) and a price per gram. A budget limits orders. A controlled compound needs an approval before it can be ordered. Tools: `search_compounds`, `get_compound`, `place_order`, `request_approval`, `cancel_order`, `get_budget`.

The 24 tasks come in three tiers of eight: single call (tier 1), lookup then act (tier 2), and policy-constrained or with near-duplicate names (tier 3).

## MCP

The tools are served over MCP v2 stdio:

```bash
python -m passk.mcp_server labstock      # or: passk serve-tools labstock
```

The server follows the 2026-07-28 MCP specification note: there are no sessions. Each episode's state is held by the server under an `env_handle` minted by `episode_reset`, and the client passes that handle explicitly on every call. Both transports give the same results for valid calls. The oracle agent's trajectories are identical in process and over MCP on all 24 tasks, and a test checks this. For calls that name an argument the tool does not declare, the two transports behaved differently before this release; see below. The `mcp` package version is recorded in each `env.json`.

**Undeclared argument names.** By default (`lenient`) both transports drop an argument name that the tool does not declare, which is what the MCP layer has always done. The E3 runs used this behaviour over MCP, so it stays the default for comparability. Setting `env_params: {strict_args: true}` rejects such names with an `invalid arguments` error on both transports. Strict mode has not been used in any experiment, so its failure counts (for example `schema_error`) are not comparable with the committed runs. The parity test covers both modes. See [`docs/deviations.md`](docs/deviations.md) (D24) and [`docs/limitations.md`](docs/limitations.md).

## CLI

| Command | What it does |
|---|---|
| `passk demo` | Scripted oracle against a noisy oracle, then a gate. SYNTHETIC. The gate verdict it prints is FAIL by design, and the exit code is 0 whatever the verdict. For a CI exit code, use `passk gate` on two run directories. |
| `passk run -c CONFIG [--smoke] [--out DIR] [--resume\|--no-resume]` | Run tasks x trials for one config and write `DIR/<run_id>/`. Finished episodes are saved as they end. `--smoke` runs one or two tasks with n=1. |
| `passk report RUN_DIR --format md\|json\|html` | Render a run report. |
| `passk diff A B` | Per-task pass^k deltas and failure shifts between two runs. |
| `passk gate A B --policy configs/gate.yaml` | Exit 0 if the candidate passes, 1 if it regresses. |
| `passk validate [--smoke] [--out DIR]` | Estimator (E1) and gate (E2) calibration. SYNTHETIC. |
| `passk serve-tools labstock` | Serve the environment over MCP stdio. |
| `passk site --results results --out build/site` | Build the static site. Writes a placeholder when no results are committed. |

Every command accepts `--json`. Exit codes: 0 pass, 1 check failed or regression, 2 usage or runtime error.

`passk lint-tools` (checks MCP tool descriptions for drift) is a stretch goal and is not implemented.

## Results

Generated by `make report` from `results/`. Do not edit this block by hand. `make check-results` fails when it is stale.

<!-- results:start -->
**Headline** (from `results/headline.json`):

- **E3 Qwen2.5-1.5B-Instruct (Q4_K_M)**: pass@1 0.383 vs pass^5 0.000, gap 0.383. 95% CI on the gap [0.250, 0.517]. 120 episodes, 24 tasks, Qwen2.5-1.5B-Instruct. n counts episodes. Failure leader: policy_violation 15.8% of episodes.
- **E3 Qwen3-1.7B Q8_0 with /no_think (optional arm)**: pass@1 0.350 vs pass^5 0.333, gap 0.017. 95% CI on the gap [0.000, 0.042]. 120 episodes, 24 tasks, Qwen3-1.7B. n counts episodes. Failure leader: tool_error_unrecovered 56.7% of episodes.
- **E5 prompt v2 vs v1 (1.5B), regression gate**: PASS. delta 95% CI [+0.117, +0.417] (delta +0.258). pass^3: baseline 0.138, candidate 0.396, delta +0.258 (95% CI [+0.117, +0.417], 24 tasks) | drop -0.258 is within max_drop 0.100 n counts the candidate run's episodes.
- **E3 Qwen2.5-0.5B-Instruct (Q4_K_M)**: pass@1 0.092 vs pass^5 0.083, gap 0.008. 95% CI on the gap [0.000, 0.025]. 120 episodes, 24 tasks, Qwen2.5-0.5B-Instruct. n counts episodes. Failure leader: tool_error_unrecovered 57.5% of episodes.
- SYNTHETIC: **E1 estimator, max absolute error of pass^k**: max abs error 0.0432. Over k = 1..5 and per-step error p in {0.00, 0.05, 0.10, 0.20}, 20 trials per task. Largest |z| 1.72; a value below 3.0 is consistent with the analytic pass^k. Calibrates the tool, not a model. No n is given: the value is a maximum over cells, not a sample size.
- SYNTHETIC: **E2 gate calibration: false-alarm rate and power**: false alarm 4.0%, power 94.5%. false alarm 95% CI [2.0%, 7.7%]; power 95% CI [90.4%, 96.9%]. Gate k=3, max_drop=0.1; 200 A/A and 200 A/B pairs at 5 trials per task. n counts the pairs in each arm. Calibrates the tool, not a model.

**SYNTHETIC: estimator and gate calibration** (scripted NoisyOracle agents; this checks the tool, not a model).

E1 (n=20 trials per task, k=1..5). Max absolute error of pass^k against the analytic value, and the largest z-score.

| per-step error p | max abs error | max abs z |
|---|---|---|
| 0.00 | 0.0000 | 0.00 |
| 0.05 | 0.0411 | 0.97 |
| 0.10 | 0.0368 | 0.97 |
| 0.20 | 0.0432 | 1.72 |

E2 gate (k=3, max_drop=0.1, 5 trials per task, 200 pairs each). Wilson 95% intervals.

| comparison | rate | count | 95% CI |
|---|---|---|---|
| false alarm, A/A (p 0.10 vs 0.10) | 4.0% | 8/200 | 2.0% to 7.7% |
| power, A/B (p 0.05 vs 0.15) | 94.5% | 189/200 | 90.4% to 96.9% |

Environment record: [`results/calibration/env.json`](results/calibration/env.json).

**E3 main run, Qwen2.5-0.5B-Instruct: `e3-qwen2.5-0.5b-t0.7`**

Agent: Qwen/Qwen2.5-0.5B-Instruct-GGUF (qwen2.5-0.5b-instruct-q4_k_m.gguf). Transport: mcp_stdio. Temperature: 0.7. Prompt: v1. 24 tasks x 5 trials = 120 episodes. Base seed 2026.

| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |
|---|---|---|
| 1 | 0.092 [0.000, 0.217] | 0.092 [0.000, 0.217] |
| 2 | 0.100 [0.000, 0.225] | 0.083 [0.000, 0.208] |
| 3 | 0.108 [0.000, 0.233] | 0.083 [0.000, 0.208] |
| 4 | 0.117 [0.000, 0.250] | 0.083 [0.000, 0.208] |
| 5 | 0.125 [0.000, 0.250] | 0.083 [0.000, 0.208] |

Failure categories, share of all episodes: loop 13.3%, max_turns 4.2%, policy_violation 1.7%, schema_error 14.2%, tool_error_unrecovered 57.5%.

Gap pass@1 - pass^5: 0.008 (95% CI [0.000, 0.025]), over tasks.

| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |
|---|---|---|---|
| 1 | 8 | 11 / 40 | 27.5% [16.1%, 42.8%] |
| 2 | 8 | 0 / 40 | 0.0% [0.0%, 8.8%] |
| 3 | 8 | 0 / 40 | 0.0% [0.0%, 8.8%] |

Tasks whose trials all took the same action sequence: 0 of 24.

Median per episode: 23.5 s wall, 310 output tokens. Totals: 611652 input and 38725 output tokens over 880 model calls. Timing was measured on a shared laptop, not a benchmark (see docs/deviations.md, D29).

[report](results/e3/e3-qwen2.5-0.5b-t0.7/report.json) | [env](results/e3/e3-qwen2.5-0.5b-t0.7/env.json)

**E3 main run, Qwen2.5-1.5B-Instruct: `e3-qwen2.5-1.5b-t0.7`**

Agent: Qwen/Qwen2.5-1.5B-Instruct-GGUF (qwen2.5-1.5b-instruct-q4_k_m.gguf). Transport: mcp_stdio. Temperature: 0.7. Prompt: v1. 24 tasks x 5 trials = 120 episodes. Base seed 2026.

| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |
|---|---|---|
| 1 | 0.383 [0.250, 0.517] | 0.383 [0.250, 0.517] |
| 2 | 0.538 [0.371, 0.704] | 0.229 [0.125, 0.338] |
| 3 | 0.600 [0.425, 0.771] | 0.138 [0.067, 0.217] |
| 4 | 0.642 [0.458, 0.817] | 0.067 [0.033, 0.108] |
| 5 | 0.667 [0.500, 0.833] | 0.000 [0.000, 0.000] |

Failure categories, share of all episodes: false_success_claim 12.5%, loop 0.8%, max_turns 10.8%, policy_violation 15.8%, schema_error 7.5%, tool_error_unrecovered 14.2%.

Gap pass@1 - pass^5: 0.383 (95% CI [0.250, 0.517]), over tasks.

| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |
|---|---|---|---|
| 1 | 8 | 30 / 40 | 75.0% [59.8%, 85.8%] |
| 2 | 8 | 11 / 40 | 27.5% [16.1%, 42.8%] |
| 3 | 8 | 5 / 40 | 12.5% [5.5%, 26.1%] |

Tasks whose trials all took the same action sequence: 0 of 24.

Median per episode: 26.4 s wall, 226 output tokens. Totals: 513611 input and 27812 output tokens over 780 model calls. Timing was measured on a shared laptop, not a benchmark (see docs/deviations.md, D29).

[report](results/e3/e3-qwen2.5-1.5b-t0.7/report.json) | [env](results/e3/e3-qwen2.5-1.5b-t0.7/env.json)

**E3 optional arm, Qwen3-1.7B (contrast with the 1.5B arm): `e3-qwen3-1.7b-q8-t0.7`**

> Optional third arm: Qwen3-1.7B, quantization Q8_0, thinking disabled (system suffix `/no_think`), context 4096 tokens. Compared with the 1.5B arm, the comparison is confounded: the model family, the quantization and the thinking mode all differ. No cause is attributed to any one of them.

Agent: Qwen/Qwen3-1.7B-GGUF (Qwen3-1.7B-Q8_0.gguf). Transport: mcp_stdio. Temperature: 0.7. Prompt: v1. 24 tasks x 5 trials = 120 episodes. Base seed 2026.

| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |
|---|---|---|
| 1 | 0.350 [0.175, 0.542] | 0.350 [0.175, 0.542] |
| 2 | 0.367 [0.183, 0.558] | 0.333 [0.167, 0.542] |
| 3 | 0.383 [0.200, 0.575] | 0.333 [0.167, 0.542] |
| 4 | 0.400 [0.208, 0.592] | 0.333 [0.167, 0.542] |
| 5 | 0.417 [0.208, 0.625] | 0.333 [0.167, 0.542] |

Failure categories, share of all episodes: false_success_claim 4.2%, policy_violation 2.5%, schema_error 1.7%, tool_error_unrecovered 56.7%.

Gap pass@1 - pass^5: 0.017 (95% CI [0.000, 0.042]), over tasks.

| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |
|---|---|---|---|
| 1 | 8 | 35 / 40 | 87.5% [73.9%, 94.5%] |
| 2 | 8 | 0 / 40 | 0.0% [0.0%, 8.8%] |
| 3 | 8 | 7 / 40 | 17.5% [8.7%, 31.9%] |

Tasks whose trials all took the same action sequence: 6 of 24.

Timing check (the pre-registered 40 s median rule, applied to this arm): the median is 17.7 s, below the threshold, so the arm ran at n = 5 as configured.

Median per episode: 17.7 s wall, 85 output tokens. Totals: 222542 input and 16297 output tokens over 396 model calls. Timing was measured on a shared laptop, not a benchmark (see docs/deviations.md, D29).

[report](results/e3/e3-qwen3-1.7b-q8-t0.7/report.json) | [env](results/e3/e3-qwen3-1.7b-q8-t0.7/env.json)

**E4 determinism control, T = 0: `e4-qwen2.5-0.5b-t0`**

Agent: Qwen/Qwen2.5-0.5B-Instruct-GGUF (qwen2.5-0.5b-instruct-q4_k_m.gguf). Transport: mcp_stdio. Temperature: 0.0. Prompt: v1. 24 tasks x 3 trials = 72 episodes. Base seed 2026.

| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |
|---|---|---|
| 1 | 0.083 [0.000, 0.208] | 0.083 [0.000, 0.208] |
| 2 | 0.083 [0.000, 0.208] | 0.083 [0.000, 0.208] |
| 3 | 0.083 [0.000, 0.208] | 0.083 [0.000, 0.208] |

Failure categories, share of all episodes: loop 20.8%, max_turns 4.2%, policy_violation 4.2%, schema_error 9.7%, tool_error_unrecovered 52.8%.

| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |
|---|---|---|---|
| 1 | 8 | 6 / 24 | 25.0% [12.0%, 44.9%] |
| 2 | 8 | 0 / 24 | 0.0% [0.0%, 13.8%] |
| 3 | 8 | 0 / 24 | 0.0% [0.0%, 13.8%] |

Tasks whose trials all took the same action sequence: 21 of 24.

Median per episode: 22.0 s wall, 303 output tokens. Totals: 355395 input and 20636 output tokens over 528 model calls. Timing was measured on a shared laptop, not a benchmark (see docs/deviations.md, D29).

[report](results/e4/e4-qwen2.5-0.5b-t0/report.json) | [env](results/e4/e4-qwen2.5-0.5b-t0/env.json)

**E5 candidate, prompt v2: `e5-qwen2.5-1.5b-v2-t0.7`**

Agent: Qwen/Qwen2.5-1.5B-Instruct-GGUF (qwen2.5-1.5b-instruct-q4_k_m.gguf). Transport: mcp_stdio. Temperature: 0.7. Prompt: v2. 24 tasks x 5 trials = 120 episodes. Base seed 2026.

| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |
|---|---|---|
| 1 | 0.592 [0.433, 0.750] | 0.592 [0.433, 0.750] |
| 2 | 0.696 [0.521, 0.863] | 0.488 [0.342, 0.638] |
| 3 | 0.708 [0.542, 0.875] | 0.396 [0.250, 0.554] |
| 4 | 0.708 [0.542, 0.875] | 0.317 [0.167, 0.483] |
| 5 | 0.708 [0.542, 0.875] | 0.250 [0.083, 0.417] |

Failure categories, share of all episodes: false_success_claim 3.3%, loop 0.8%, max_turns 5.0%, policy_violation 20.8%, schema_error 5.0%, tool_error_unrecovered 5.8%.

Gap pass@1 - pass^5: 0.342 (95% CI [0.192, 0.492]), over tasks.

| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |
|---|---|---|---|
| 1 | 8 | 32 / 40 | 80.0% [65.2%, 89.5%] |
| 2 | 8 | 29 / 40 | 72.5% [57.2%, 83.9%] |
| 3 | 8 | 10 / 40 | 25.0% [14.2%, 40.2%] |

Tasks whose trials all took the same action sequence: 0 of 24.

Median per episode: 21.9 s wall, 187 output tokens. Totals: 392861 input and 25922 output tokens over 682 model calls. Timing was measured on a shared laptop, not a benchmark (see docs/deviations.md, D29).

[report](results/e5/e5-qwen2.5-1.5b-v2-t0.7/report.json) | [env](results/e5/e5-qwen2.5-1.5b-v2-t0.7/env.json)

**E6 descriptive robustness run (injected transient tool failures): `e6-qwen2.5-1.5b-flaky0.2`**

> Descriptive robustness run: each tool call fails transiently with probability 0.2, seeded per episode. This run has 3 trials per task, and the E3 1.5B baseline has 5 trials per task, so the two are compared descriptively only. No significance claim is made.

Agent: Qwen/Qwen2.5-1.5B-Instruct-GGUF (qwen2.5-1.5b-instruct-q4_k_m.gguf). Transport: mcp_stdio. Temperature: 0.7. Prompt: v1. 24 tasks x 3 trials = 72 episodes. Base seed 2026.

| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |
|---|---|---|
| 1 | 0.403 [0.264, 0.542] | 0.403 [0.264, 0.542] |
| 2 | 0.569 [0.389, 0.750] | 0.236 [0.111, 0.375] |
| 3 | 0.625 [0.417, 0.792] | 0.125 [0.000, 0.250] |

Failure categories, share of all episodes: false_success_claim 4.2%, loop 1.4%, max_turns 4.2%, policy_violation 8.3%, schema_error 5.6%, tool_error_unrecovered 36.1%.

| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |
|---|---|---|---|
| 1 | 8 | 17 / 24 | 70.8% [50.8%, 85.1%] |
| 2 | 8 | 7 / 24 | 29.2% [14.9%, 49.2%] |
| 3 | 8 | 5 / 24 | 20.8% [9.2%, 40.5%] |

Recovery (the pre-registered E6 measure): 17 of 53 episodes with at least one transient failure still succeeded (Wilson 95% CI [21.1%, 45.5%]). Transient failures recorded: 81.

Tasks whose trials all took the same action sequence: 0 of 24.

Median per episode: 25.1 s wall, 225 output tokens. Totals: 293962 input and 16949 output tokens over 468 model calls. Timing was measured on a shared laptop, not a benchmark (see docs/deviations.md, D29).

[report](results/e6/e6-qwen2.5-1.5b-flaky0.2/report.json) | [env](results/e6/e6-qwen2.5-1.5b-flaky0.2/env.json)

**E7b descriptive hosted run (3 tasks): `e7b-nemotron-3-ultra-stratified-t0.7`**

> Descriptive only: 3 tasks (the first of each tier), 3 trials each, 9 episodes. This supports no general claim about the model. Provider-reported model id: `nvidia/nemotron-3-ultra-550b-a55b:free`. Free-tier routing and model versions can change between days.

Agent: nvidia/nemotron-3-ultra-550b-a55b:free (reported: nvidia/nemotron-3-ultra-550b-a55b:free). Transport: mcp_stdio. Temperature: 0.7. Prompt: v1. 3 tasks x 3 trials = 9 episodes. Base seed 2026.

| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |
|---|---|---|
| 1 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 2 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 3 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |
|---|---|---|---|
| 1 | 1 | 3 / 3 | 100.0% [43.9%, 100.0%] |
| 2 | 1 | 3 / 3 | 100.0% [43.9%, 100.0%] |
| 3 | 1 | 3 / 3 | 100.0% [43.9%, 100.0%] |

Tasks whose trials all took the same action sequence: 3 of 3.

Median per episode: 17.0 s wall, 596 output tokens. Totals: 28563 input and 5592 output tokens over 48 model calls. Timing was measured on a shared laptop, not a benchmark (see docs/deviations.md, D29).

[report](results/e7b/e7b-nemotron-3-ultra-stratified-t0.7/report.json) | [env](results/e7b/e7b-nemotron-3-ultra-stratified-t0.7/env.json)

**Descriptive comparison: E6 against the E3 1.5B baseline.** Not a significance test: the trial counts differ, and the two runs are not paired.

| run | trials per task | successes / episodes | pass@1 (95% CI) | pass^3 (95% CI) |
|---|---|---|---|---|
| `e3-qwen2.5-1.5b-t0.7` | 5 | 46 / 120 | 0.383 [0.250, 0.517] | 0.138 [0.067, 0.217] |
| `e6-qwen2.5-1.5b-flaky0.2` | 3 | 29 / 72 | 0.403 [0.264, 0.542] | 0.125 [0.000, 0.250] |

**Descriptive contrast: the Qwen3-1.7B arm against the E3 1.5B baseline.** The comparison is confounded: the model family, the quantization and the thinking mode all differ between the rows. No cause is attributed.

| run | model | quantization | thinking | pass@1 (95% CI) | pass^5 (95% CI) |
|---|---|---|---|---|---|
| `e3-qwen2.5-1.5b-t0.7` | Qwen/Qwen2.5-1.5B-Instruct-GGUF | Q4_K_M | not applicable (no thinking mode) | 0.383 [0.250, 0.517] | 0.000 [0.000, 0.000] |
| `e3-qwen3-1.7b-q8-t0.7` | Qwen/Qwen3-1.7B-GGUF | Q8_0 | disabled (`/no_think`) | 0.350 [0.175, 0.542] | 0.333 [0.167, 0.542] |

Pre-registered E5 comparison: baseline E3 1.5B (prompt v1), candidate E5 (prompt v2). Gate output, [`results/gates/e5_v1_vs_v2.txt`](results/gates/e5_v1_vs_v2.txt):

```
PASS
  - pass^3: baseline 0.138, candidate 0.396, delta +0.258 (95% CI [+0.117, +0.417], 24 tasks)
  - drop -0.258 is within max_drop 0.100
```

POST-HOC demonstration, NOT pre-registered: the same two E5 runs with the roles swapped (baseline prompt v2, candidate prompt v1). It shows the gate firing on a drop; it is not a second E5 result. Gate output, [`results/gates/e5_reversed_v2_baseline_v1_candidate.txt`](results/gates/e5_reversed_v2_baseline_v1_candidate.txt):

```
FAIL
  - pass^3: baseline 0.396, candidate 0.138, delta -0.258 (95% CI [-0.417, -0.117], 24 tasks)
  - regression: drop 0.258 exceeds max_drop 0.100 and the 95% CI excludes 0
```
<!-- results:end -->

## Findings

Every number is in the results block above, or in `results/headline.json`. This section only says what the numbers show.

**E3, two models, 24 tasks x 5 trials, prompt v1.** Each model has its own section in the block: pass@k and pass^k for k = 1 to 5 with bootstrap intervals, the gap between pass@1 and pass^5, a tier table with Wilson intervals, and failure categories as a share of all episodes. For the 1.5B model, one attempt succeeds far more often than every attempt does, and the gap widens as k grows. That gap is what this harness is built to show. The 0.5B model's successes sit mostly in tier 1, and its pass@k and pass^k curves nearly coincide, so there is little gap to show.

**E4, the same 0.5B model at T = 0 (greedy) with a fixed base seed, 3 trials per task.** Across the three trials, success was identical for every task. Three of the 24 tasks differed in their action sequences: `t1_01` (the same `grams` value written as `25` in one trial and `25.0` in another), `t3_04` (the first action names `200` as the compound in one trial and `Acetonitrile` in the other two), and `t3_07` (a schema error in one trial, where the other two produced a valid call). The failure category of `t3_07` also differed between trials. The cause of the non-identical trials was not investigated. The block shows how many tasks took the same action sequence in all trials. In this run pass@1 and pass^3 are equal, because no task mixes success and failure.

**E5, the 1.5B model with prompt v2 against v1, the pre-registered gate (`results/gates/e5_v1_vs_v2.txt`).** The verdict is PASS in the pre-registered direction: the v2 run's pass^3 is above the v1 run's, with an interval that excludes zero. Stated plainly, the result is surprising. Removing the policy section from the prompt improved pass^3 for this model, and more tasks succeed under v2. The policy-violation share is higher under v2 than under v1 (see both failure lines in the block), so the improvement is not a reduction in policy violations. This experiment measured what the prompt change did to this model's results. It did not test why. Two untested hypotheses: the policy text may distract the model from the tool arguments, or the policy text may change how the model orders its calls. Testing either would need a new experiment. Four v2 trajectories were read by hand (`docs/failure-analysis.md`).

**Reversed gate, a post-hoc demonstration only.** `results/gates/e5_reversed_v2_baseline_v1_candidate.txt` runs the same two E5 runs with the roles swapped, so the v2 run is the baseline. The gate reports FAIL, because the drop exceeds the policy's limit. This was not pre-registered. It shows the gate firing on a drop. It is not a second E5 result, and it is not evidence that prompt v1 is better than v2.

**E6, injected transient tool failures (1.5B, prompt v1, 3 trials per task).** A descriptive robustness run. Each tool call fails transiently with probability 0.2. The block shows its pass@k and pass^k beside the E3 1.5B baseline, which has more trials per task, and the pre-registered recovery measure: among the episodes with at least one transient failure, a minority still succeeded. Two trajectories read by hand show the model retrying some transient failures and not others. There is no significance claim, because the trial counts differ and the runs are not paired.

**Qwen3-1.7B, the optional third arm (Q8_0, thinking disabled with `/no_think`, context 4096).** Run at the pre-registered size (24 tasks, 5 trials per task). Its pass@1 and pass^5 are close, so the gap that the 1.5B arm shows is small here. The comparison with the 1.5B arm is confounded: the model family, the quantization (Q8_0 against Q4_K_M) and the thinking mode all differ, so no cause is attributed. The 40 s timing check was applied to this arm and was not triggered.

**E7b, hosted free model, descriptive (3 tasks, the first of each tier, 3 trials each).** Every episode succeeded. That supports no general claim about the model: three tasks are too few. The provider-reported model id is in the block. Free-tier routing and model versions can change between days. The full E7 design was not run because the free tier allows 50 model requests per day per account, and that quota does not fit it.

**Timing.** The wall times in the block are measured on a shared laptop, not as a benchmark. The 0.5B E3 run overlapped with other jobs for part of its run, so its times are not comparable with the 1.5B run (`docs/deviations.md`, D29).

## Experiments

Pre-registered in [`docs/prereg.md`](docs/prereg.md) before any result file existed. Deviations are logged in [`docs/deviations.md`](docs/deviations.md). Failure reading is in [`docs/failure-analysis.md`](docs/failure-analysis.md): 13 quoted trajectories from the E3 runs, read by hand. The quoted sequences were re-checked against the trajectory files.

| ID | Question | Status |
|---|---|---|
| E1 | Does the estimator track the analytic pass^k? | Run (SYNTHETIC) |
| E2 | What are the gate's false-alarm rate and power? | Run (SYNTHETIC) |
| E3 | pass@k and pass^k for Qwen2.5-0.5B and 1.5B | Run, complete (120 episodes each) |
| E4 | Does T=0 remove the pass^k gap? | Run, complete (72 episodes). See Findings. |
| E5 | Does prompt v2 regress against v1 on the 1.5B model? | Run, complete. Pre-registered gate verdict: PASS in the pre-registered direction. See Findings. |
| E6 | Recovery from transient tool failures (stretch, descriptive) | Run, complete (3 trials per task). See Findings. |
| E7 | Hosted free model arm, full design (24 tasks x 3 trials) | Not run: the free tier allows 50 model requests per day per account, and the full design does not fit that quota. |
| E7b | Hosted arm, stratified (3 tasks x 3 trials, descriptive) | Run, complete. See Findings. |
| Qwen3-1.7B arm (optional third model, Q8_0) | Run, complete. A confounded contrast with the 1.5B arm. See Findings. |

## Hosted arm

The hosted arm (`openai_compat` agent, OpenRouter, `nvidia/nemotron-3-ultra-550b-a55b:free`) is implemented and tested: the HTTP layer is covered with a fake post, and a one-task live smoke run succeeded. A stratified descriptive run, E7b (3 tasks x 3 trials), completed on the free quota. The full design (E7, 24 tasks x 3 trials) was **not run**: it needs more free requests per day than the account allows, and an attempt stopped on the daily quota. A full evaluation needs an account with more than 50 free requests per day. `docs/deviations.md` (D21, D22, D34) and `docs/prereg.md` (amendments 2 and 3) say the same thing.

## Interrupted and partial runs

Each finished episode is written to disk as it ends. If a run is interrupted, re-running the same command continues it: `passk run --resume` is the default when the configuration matches, and a different configuration is refused. If a provider quota stops a run, the finished episodes are kept, a partial report is written with `complete: false`, and the CLI exits with code 2. The message gives the reset time when the provider sends one. Partial reports are labelled PARTIAL in the site, the markdown report and the results block below, and `passk gate` refuses them.

## Design

Decisions and their reasons are in [`docs/design.md`](docs/design.md). The action schema, the gate definition and the calibration design are described there.

## What this does not show

See [`docs/limitations.md`](docs/limitations.md). In short: one domain, two small models, single-instruction episodes, no judge, a small sample per task, and hardware limited to one laptop.

## Reproduce

```bash
make setup          # uv sync --extra dev (no llama-cpp-python)
make test           # tests, model tests skipped
make check-results  # README block and results/headline.json match results/
make dogfood        # validate --smoke and two gate checks on fixtures
```

Local model runs need the native build: `uv pip install -e '.[llama]'`. Then, for example:

```bash
passk run -c configs/e3_qwen05.yaml --smoke          # setup check, about 40 s
passk run -c configs/e3_qwen05.yaml --out results/e3 # main run, long
passk run -c configs/e3_qwen3_17b.yaml --out results/e3  # optional Qwen3-1.7B arm
passk run -c configs/e6_flaky.yaml --out results/e6     # E6, transient tool failures
# E7b uses the free hosted quota. Set OPENROUTER_API_KEY in the environment first; the key is never stored.
passk run -c configs/e7b_hosted_stratified.yaml --out results/e7b
```

## Licences

- passk: MIT (see [`LICENSE`](LICENSE)).
- LabStock tasks and environment: written for this repository. They use no external dataset.
- Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (GGUF, Q4_K_M): the licence is not verified in this build. Check the model pages on Hugging Face (Qwen/Qwen2.5-0.5B-Instruct-GGUF and Qwen/Qwen2.5-1.5B-Instruct-GGUF). The Qwen3-1.7B arm is not run.
- Hosted free model (`nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter): see the provider page.

## How this was built

This repository was designed, coded, tested and documented by AI coding assistants (Claude, run through Claude Code) working autonomously under Ayush Kumar's direction. That includes the pre-registrations, the experiments and this write-up. A separate AI reviewer checked the code and the results (the findings and what was changed are in the commit history and `docs/deviations.md`). A line-by-line human review by the author has not been done yet, so none of this should be read as a human-audited result. Every measured number in this README is generated from the committed result files.

## Citation

See [`CITATION.cff`](CITATION.cff).

<!-- markers are rendered by scripts/render_readme.py -->

## Continuous integration

The CI workflow is `.github/workflows/ci.yml`. The Pages workflow (`pages.yml`) stays in `ci-templates/`, because the site is deployed from the `gh-pages` branch.
