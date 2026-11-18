# Pre-registration

Written 2026-11-19, committed before any file exists in `results/`. Any change to n, seeds,
prompts, thresholds, models or stop rules after this commit is logged in `docs/deviations.md`
with a reason. No expected numbers are written here: the study is reported whatever it finds.

Development runs of `passk validate` were made before this commit to find the bugs in
`docs/deviations.md`. Their outputs are not reported. The committed calibration (E1, E2) is
produced by a run made after this commit, with the seeds below.

## Common settings

- **Tasks:** the 24 packaged LabStock tasks (`src/passk/tasks/labstock/*.yaml`), tiers 1 to 3,
  8 each.
- **Seeds:** E1 uses base seed 0 + noise index. E2 uses base seed 0; A/A pair i uses seeds
  `10000 + 2i` and `10000 + 2i + 1`, A/B pair i uses `20000 + 2i` and `20000 + 2i + 1`. E3, E4,
  E5 and E7 use base seed 2026. Episode seeds are `episode_seed(base, task, trial)`, a SHA-256
  of the three values.
- **Intervals:** Wilson 95% for proportions. Percentile bootstrap over tasks for pass@k and
  pass^k, 10,000 resamples, seed 0. Paired bootstrap over tasks for gate deltas. No
  multiplicity correction. A claim of a difference is made only when its interval excludes 0.
- **Gate policy** (`configs/gate.yaml`): k = 3, max_drop = 0.10, significance required,
  alpha = 0.05. The policy is fixed before any model run.
- **Hardware and runtime:** Intel i7-4770HQ laptop (4 cores, 8 threads), CPU only,
  `n_threads` = 4, `n_gpu_layers` = 0, `n_ctx` = 4096, `max_tokens` = 256. Model results are
  Qwen2.5 models and are not generalised to other models or hardware.
- **Transport:** MCP stdio (`python -m passk.mcp_server labstock`) for every model run.
- **Recorded per run:** `env.json` (git SHA, CPU, Python and package versions, model
  repo_id, file name and SHA-256, `n_threads`, seed, AC or battery state).

## E1. Estimator validation (no model)

- **Procedure:** `passk validate`. NoisyOracle agents with per-step error p in
  {0, 0.05, 0.10, 0.20}, 20 trials per task. For each level and k = 1..5, compare measured
  pass^k (mean over tasks of the unbiased estimator) with the analytic value
  mean_t (1 - p)^(m_t k), where m_t is the reference length.
- **Measures:** the max absolute error over k, per level; and the z-score of each cell, the
  error divided by its exact standard error under the model.
- **Pass criterion:** max |z| < 3.0 at every level. Errors are reported in any case.
- **Stop rule:** none.

## E2. Gate calibration (no model)

- **Procedure:** `passk validate`. 200 A/A pairs (p = 0.10 against p = 0.10) and 200 A/B pairs
  (p = 0.05 baseline against p = 0.15 candidate). 5 trials per task, the same n as E3, at
  k = 3 with `configs/gate.yaml`.
- **Measures:** false-alarm rate on A/A pairs and detection rate (power) on A/B pairs, each
  with a Wilson 95% interval.
- **Decision rule:** if the upper bound of the A/A interval exceeds 0.10 (twice the nominal
  alpha), the gate is reported as anti-conservative. The gate is not retuned in this study.
  The power figure is reported as it is, with no threshold.

## E3. Main run (model)

- **Configs:** `configs/e3_qwen05.yaml` and `configs/e3_qwen15.yaml`.
- **Models:** Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, Q4_K_M GGUF, Apache-2.0.
- **Settings:** T = 0.7, prompt v1, 24 tasks x 5 trials, MCP stdio.
- **Measures:** pass@1 and pass@5; pass^1 to pass^5, each with a bootstrap 95% interval;
  failure categories by tier; median input and output tokens and median wall time per episode.
- **Timing rules, decided from the smoke runs before the full run (DECISION.md section 2.1):**
  if the median 1.5B episode takes more than 40 s, the 1.5B arm runs with n = 3; if more than
  90 s, the 1.5B arm is dropped. Either change is logged in `docs/deviations.md`.
- **Prompt rule:** if both models score 0 on tier 1 in the smoke run, the prompt or the schema
  is fixed once and the change is logged. If they still score 0, the failure taxonomy is the
  reported result.
- **Qwen3-1.7B arm (optional):** added only if a measured decode speed of at least 8 tok/s is
  recorded before the run. The measurement and its outcome are logged, whichever way it goes.

## E4. Determinism control (model)

- **Config:** `configs/e4_determinism.yaml`. Qwen2.5-0.5B, T = 0 (greedy), 3 trials per task.
- **Measures:** for each task, whether all three action sequences are identical; pass@1
  against pass^3.
- **Interpretation:** if the sequences are identical, pass^3 should equal pass@1. Any gap that
  remains is reported as non-determinism in the runtime and not as agent behaviour.
- **Stop rule:** none.

## E5. Prompt regression check (model)

- **Config:** `configs/e5_prompt_v2.yaml`. Qwen2.5-1.5B with prompt v2 (terse, no policy
  section), T = 0.7, 5 trials per task.
- **Baseline:** E3 1.5B with prompt v1.
- **Procedure:** `passk gate` on the two run directories with `configs/gate.yaml`, plus
  `passk diff` for per-task changes. The verbatim gate output is committed. The gate's verdict
  is the result, whatever it is.
- **Note:** the two arms share base seed 2026, but the prompts differ, so the trajectories
  are not paired at the step level.

## E6. Recovery from transient failures (stretch, not run)

- **Planned design:** 0.5B agent with `flaky_p` = 0.2. Recovery rate = the share of episodes
  with at least one transient failure that still succeed.
- **Status:** not runnable yet. `flaky_p` is not settable from `RunConfig`, and the schema
  is frozen. It needs a schema change approved by the orchestrator. If it is run later, this
  section is updated first and the change is logged.

## E7. Hosted free model arm (descriptive, model)

- **Config:** `configs/e7_hosted_nemotron.yaml`. Model `nvidia/nemotron-3-ultra-550b-a55b:free`
  via OpenRouter (`openai_compat` agent), T = 0.7, 3 trials per task, `min_interval_s` = 3.
- **Key:** read from the `OPENROUTER_API_KEY` environment variable. It is never written to a
  file in this repository, a result or a log.
- **Smoke check:** one tier-1 and one tier-2 task at n = 1 before the full run.
- **Measures:** pass@1 and pass^3 with bootstrap intervals, failure categories, and the
  provider-reported model id, recorded per run.
- **Caveats, stated with the results:** free-tier endpoints are not deterministic even at a
  fixed seed; rate limits can abort a run. A run aborted by a provider error writes nothing
  and is rerun from the start, and the number of aborted attempts is reported. Provider model
  versions drift, so each result carries its date and the reported model id.
- **Not claimed:** E7 is not a gate comparison and not a significance test against E3. It is
  not comparable to the local models except as a descriptive reference.

## Not pre-registered

Changes to n, the prompt set, the tasks, the models, the gate threshold and the seeds that
are not listed above are deviations. Each is logged in `docs/deviations.md` with its reason and
the run it affected.
