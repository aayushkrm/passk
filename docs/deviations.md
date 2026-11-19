# Deviations

Every change of n, prompt, threshold, model, seed or stop rule is logged here with a reason.
Entries are dated and say which runs they affect. Entries about code bugs say what was fixed
and how it was checked.

## Before the pre-registration commit (development)

### D1. Estimator error from correlated trials (fixed)

- **Symptom:** the first E1 run gave max absolute errors of pass^k of 0.33 to 0.48 against
  the analytic values, far outside sampling noise for 20 trials per task.
- **Cause:** the NoisyOracle draws did not depend on the episode. Every trial of a task
  replayed the same corruption pattern, so the trials of a task were identical and the
  estimator saw one sample per task.
- **Fix:** `NoisyOracleAgent.begin_episode(seed)` mixes the runner's per-episode seed into the
  draws, and the runner calls it before each episode (the call was already in the initial code; now covered by
  the calibration checks).
- **Check:** after the fix, E1 errors are within sampling noise (z-scores in the calibration
  file, `results/calibration/validate.json`). Per-task success under p = 0.2 with 200 trials
  matches (1 - p)^m_t within noise (largest |z| about 1.2 over the 24 tasks in a direct check,
  not committed). With p = 1 no episode succeeds.

### D2. E2 at 5 trials per task instead of 20

- **Change:** E2 (gate calibration) runs 5 trials per task. E1 keeps 20.
- **Reason:** with 20 trials per task the 800 simulated runs took too long for the calibration
  to be a routine check. E2 at 5 trials matches the main experiment (E3, n = 5), so the gate is
  calibrated at the operating point it is used at.
- **Honesty note:** this choice was made after a development run with 2 pairs at 20 trials. No
  200-pair run at 20 trials was reported, and E2 at 20 trials was not evaluated.

### D3. Calibration speed-ups that do not change results

- Simulations skip disk writes, span building, the environment record and the summary bootstrap
  (`runner.run(..., light=True, write=False)`).
- Task files are parsed once per process instead of once per run (parsing cost about a second
  per run).
- The tool list is fetched once per run instead of once per episode.
- Check: the E1 errors from development runs were identical before and after these changes,
  with identical seeds.

### D4. Scripted runs were not labelled SYNTHETIC (fixed)

- `runner.run` did not set `env_info.synthetic` for `oracle`, `noisy_oracle` or `replay`. The
  diff, gate and site code read that flag, so their SYNTHETIC banners never appeared. Fixed:
  scripted agent kinds set the flag.

### D5. Action schema allows a missing tool name (not changed; flagged)

- The flat action schema (`passk.agents._common.action_json_schema`) requires only `kind`.
  `tool` is an enum but is not required, so the grammar allows an action with no tool name. The
  parser then records a schema error.
- **Observed:** in a direct check of ten calls with the 0.5B model, four outputs had no usable
  tool name or had `null` arguments. The two-task smoke run ended with schema errors on both
  tasks.
- **Decision (superseded by D14):** the decision was to make all four keys required
  before any model run. See D14.

## At and after the pre-registration commit

### D6. E6 cannot run

- `flaky_p` is set on `LabStockEnv`, but `RunConfig` has no field for it, and the schema is
  frozen. E6 stays in the pre-registration as a stretch and is not run. A schema change, with its
  own recorded decision, is needed first.

### D7. Qwen3-1.7B arm not added

- The optional third arm needs a measured decode speed of at least 8 tok/s. No spike record
  exists in the repository (`spike/spike.json` is absent), and no measurement was made here. The
  arm is not added. The decision is recorded here and not in a config.

### D8. Qwen2.5-1.5B weights are not on disk

- `~/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/` holds only a download cache. The E3 1.5B config,
  the E5 config and the 1.5B arm of the timing rule cannot run until the Q4_K_M file is present.
  The configs assume `~/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf`.

### D9. Timing rule for the 1.5B arm (applied; not triggered)

- The prereg's timing rule (`docs/prereg.md`, E3) asks for n = 3 on the 1.5B arm if its median
  episode takes more than 40 s, and drops the arm above 90 s. The check was made on the completed
  E3 runs. No 1.5B smoke run was made before the full run, so the prereg's "decided from the smoke
  runs" wording was not followed for this arm (D20 said otherwise; it is corrected there).
- The medians of wall time per episode, from the committed trajectories (the results block gives
  the same numbers), are 23.5 s for the 0.5B run and 26.4 s for the 1.5B run. Both runs were on a
  shared laptop (D29). Both are below 40 s, so neither rule triggered, and the 1.5B arm ran at
  n = 5 as configured.

### D10. Model SHA-256 and repo id in env.json (added)

- `LlamaCppAgent` records `repo_id`, the file name and the SHA-256 of the model file in the
  run metadata, so `env.json` carries them. Runs made before this
  change, if any, do not carry them. None exist in `results/`.

### D11. Licence of the models not verified in this build

- The Qwen GGUF files carry no licence field in their metadata, and the model card text that
  could be fetched did not include the licence front matter. The README gives the licence as
  "see the model page" and does not assert Apache-2.0 until it is checked.
- The hosted free model's licence is not verified. The README says "see provider page".

### D12. Calibration result file written after the pre-registration

- `docs/prereg.md` was committed before `results/calibration/` was created. Development runs of
  `passk validate` were made before that commit (see D1 to D3). The committed calibration is the
  output of a run made after the commit, with the same seeds, and its command is recorded in the
  README.

### D13. Hosted smoke

- A one-task smoke run (`configs/e7_hosted_nemotron.yaml`, `--smoke --smoke-tasks 1`) completed
  against the hosted endpoint. The provider-reported model id matched the requested id. The key
  was read from the environment and not written anywhere. This is a setup check, not a result,
  and its output is not in `results/`.

## Pre-run amendments (decisions made before any model result)

All of these are recorded in `docs/prereg.md`, Amendment 1.

### D14. Action schema: four required keys (supersedes D5)

- **Change.** `action_json_schema` requires `kind`, `tool`, `arguments` and `message`. `tool`
  is an enum of the tool names plus `"none"`, which a final action uses. `arguments` is `{}`
  for a final action and `message` is empty for a tool call. The parser checks the values, as
  before. History rendering uses the same four keys.
- **Measured, first action only, 20 calls with the 0.5B model, prompt v1, temperature 0.7.**
  Before (HEAD schema and prompt): 11 of 20 were schema errors. After: 0 of 20. The check is in
  scratch and is not committed. It covers the first action of each episode only.
- **Smoke (two tasks, n = 1, 0.5B).** Under the new schema, 0 of 12 steps were schema errors.
  The failures are now semantic: wrong compound ids and repeated calls. Both tasks still
  failed, which is a result of the model, not of the schema.

### D15. Prompt v2 is v1 without the policy section

- **Change.** v2 used to shorten the tool descriptions as well. The earlier plan described that
  as part of v2. It was then decided that v2 should differ from v1 only by the policy section, so
  v2 now keeps the full descriptions. v1 changed only in the text that describes the four fields.
- **Test.** `tests/test_llamacpp_agent.py` checks that v2 equals v1 with the policy block
  removed.

### D16. `RunConfig.env_params` and the flaky-transient E6 arm

- **Change.** `RunConfig` has a new field, `env_params` (default `{}`), backwards compatible with
  existing configs. For labstock, `flaky_p` is the only accepted key. It is validated by
  `validate_env_params`. The in-process environment receives it directly. The MCP server receives
  it as a JSON argument (`--env-params`). `schemas/run_report.schema.json` is regenerated.
- **E6.** `configs/e6_flaky.yaml` runs Qwen2.5-1.5B with `flaky_p` = 0.2, n = 3. The stretch is
  now runnable. It has not been run.
- **Test.** `tests/test_env_params.py` checks that `flaky_p = 1.0` makes every call fail once and
  the oracle recovers, and that the MCP path gives the same trajectories as the in-process path.

### D17. Failure counts count unsuccessful trials only

- **Change.** `TaskStats.failures` and `RunReport.failure_totals` count unsuccessful trials. The
  runner now computes them with `metrics.aggregate_task` and the summary with
  `metrics.summarize_tasks`. The duplicate `runner._aggregate` logic is gone. The site failure
  chart does not show `success`. The tests that asserted the old counts were updated, and a
  runner test compares a written report with the helpers.
- **Effect on results.** None: no model result exists yet.

### D18. Code hygiene from the review (no change to experiments)

- Reports store the task glob relative to the repository, the working directory or the installed
  package. Absolute globs are not stored. A test checks that no committed result contains a user
  path.
- `envinfo` moved into the package. `scripts/envinfo.py` is a thin wrapper. The installed wheel
  carries the full environment record, and a test builds a wheel and checks it.
- The MCP server subprocess receives only `PATH`, `PYTHONPATH`, `HOME`, `LANG` and
  `VIRTUAL_ENV`, on top of the SDK's default set. A test checks that `OPENROUTER_API_KEY` is not
  visible to the child.
- The gate refuses runs with different trial counts per task. The local bootstrap fallback is
  removed.
- Provider errors are raised outside their `except` blocks, so no transport error or response
  text is chained into them.
- The `passk demo` help says it exits 0 on a FAIL verdict.
- `scripts/render_readme.py --check` was confirmed to fail on a stale README block.

### D19. Qwen3-1.7B arm added

- **Decode check.** Qwen3-1.7B Q8_0 with 4 threads, a short benchmark outside the repository
  (two timed calls after a warm-up, about 40 prompt tokens, `/no_think`). The median decode
  speed cleared the 8 tok/s rule in `docs/prereg.md`. The number is in the report and not in the
  repository.
- **Change.** `LlamaCppAgent` gained `system_suffix` (default empty), appended to the system
  message. The Qwen3 config sets it to `/no_think`. `configs/e3_qwen3_17b.yaml` is the optional
  third arm. Its `repo_id` is `Qwen/Qwen3-1.7B-GGUF`.

### D20. Qwen2.5-1.5B weights are on disk (closes D8)

- The Q4_K_M file is in `~/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/`. The E3 1.5B, E5 and E6
  configs can run. D9 records the timing-rule check. No 1.5B smoke run was made; the check uses
  the completed E3 medians instead (see D9).

## Hosted quota and resumable runs (after the first E7 attempt)

### D21. E7 not completed at the pre-registered size (free-tier quota)

- **What happened.** The first full E7 attempt (`configs/e7_hosted_nemotron.yaml`, 24 tasks x 3
  trials) stopped on the provider's daily quota: the free tier allows 50 model requests per day
  per account, and the account had used them up. Nothing was written, because the runner then
  discarded the partial run on any provider error.
- **Decision.** E7 stays in the pre-registration as optional and quota-limited. The full size
  may need more than one day on the free tier. Amendment 2 in `docs/prereg.md` makes a partial
  E7 report reportable with `complete: false`. E7 was not run again on the day of the failure.
- **Status.** Not complete. The hosted arm is implemented and tested (mocked tests, and a
  one-task live smoke run that succeeded). Its full evaluation needs an account with more than
  50 free requests per day.

### D22. Incremental persistence, resume and the quota stop

- **Change (runner).** Each finished episode is appended to `trajectories.jsonl` and its spans
  to `spans.jsonl`, and each write is fsynced before the next episode starts. `state.json`
  records the configuration hash, the status and the progress.
- **Resume.** `passk run --resume` is the default when the run directory holds a run of the same
  configuration and task set. Finished `(task, trial)` pairs are skipped. A torn last line (an
  interrupted write) is cut before the run continues. A different configuration is refused with
  a clear error. A complete run is not overwritten. `--no-resume` refuses any existing run.
- **Quota stop.** A provider error (a quota error, or retries used up) ends the run cleanly. The
  finished episodes are kept, `report.json` is written with `complete: false`, `stop_reason` and
  `reset_at`, and the CLI exits with code 2. The message gives the reset time when the provider
  sent one. The episode that was running is discarded and rerun on resume. A crash (not a
  provider error) is not caught: the files written so far stay valid for a resume.
- **Hosted agent.** A 429 whose quota is exhausted (`X-RateLimit-Remaining: 0`, or a reset more
  than 300 s away) is not retried. The reset time is read from `Retry-After` or
  `X-RateLimit-Reset`.
- **Renderers and gate.** Partial reports carry a PARTIAL banner in the site, the markdown report
  and the README results block. `passk gate` refuses an incomplete report with exit code 2.
  `passk diff` prints a warning.
- **Schema.** `RunReport` gained `complete`, `stop_reason`, `reset_at`, `episodes_total` and
  `episodes_done`, all with defaults, so older reports still load.
- **Effect on results.** None on the existing results, which are calibration output only. The
  byte-identical resume test covers trajectories and spans.

### D23. Headline contract (hub standard)

- `results/headline.json` now follows the hub's standard contract (`schema_version` "1"). It
  replaces the earlier internal format, and nothing else read that format. Contract and schema are
  in `schemas/headline.schema.json`, and `tests/test_headline.py` checks them.
- **Six-item cap.** The failure-category leader is not a separate item. It is part of each E3
  item's `detail`, because the separate items would push the E5 verdict or the E7 partial out of
  the cap.
- **E5 verdict.** It comes from `passk.gate.gate` on the two report JSON files, not from the
  `results/gates/*.txt` file. The text file is still shown verbatim in the README block.
- **Optional arm.** The Qwen3-1.7B item is the lowest priority, so it is the one dropped when the
  cap is reached.

## Review 2 fixes: transports, provenance and documentation

### D24. E3 ran lenient over MCP; the earlier parity claim covered valid calls only

- **What ran.** The E3 runs (both models) used the MCP stdio transport. The MCP layer drops argument
  names that a tool does not declare before the tool runs, so an undeclared name was silently
  ignored. The in-process environment rejected the same call with `invalid arguments`. The two
  transports therefore differed on undeclared names. The README's "identical trajectories" claim
  was true for valid calls only. Two recorded E3 trajectories show the effect: 0.5B `t1_07`
  trial 0 (a `get_compound` call with an extra `arguments` key, accepted) and 0.5B `t3_01` trial 4
  (a `place_order` with an extra `reason` key, accepted and placed). Both were found by the second
  review (REVIEW-2, item R-1).
- **Change.** Lenient is now the default on both transports. The in-process environment drops
  undeclared names, as the MCP layer always did, so the E3 results and the E5 run stay comparable.
  A new opt-in env parameter, `strict_args` (default false), rejects undeclared names on both
  transports. Under strict mode the MCP server tightens the argument model of the environment tools
  only. The control tools are unchanged. The two transports' error texts differ (labstock's message
  in process, the SDK's validation message over MCP), and both name the undeclared key.
- **Tests.** `tests/test_strict_args.py` checks lenient parity across the transports, strict
  rejection on both, and that valid calls give the same results in strict mode. The E1 and E2
  calibrations are unaffected, because their agents never send undeclared names.
- **Effect on results.** None on the completed E3 runs, which were lenient over MCP. E5 and E4 run
  under the same lenient behaviour. Strict mode has not been used in any experiment.

### D25. Bootstrap seed: 2026 in the code, 0 in the prereg

- The prereg (`docs/prereg.md`, Common settings) says the pass@k and pass^k bootstrap uses seed 0.
  The runner computes the run summary with `metrics.summarize_tasks(..., seed=config.base_seed)`,
  so the E3 confidence intervals use seed 2026, the run's base seed. The gate and the calibration use
  seed 0, as the prereg says.
- The reported E3 intervals are internally consistent: they match an independent recomputation
  with seed 2026. The code was not changed, because that would change the completed results.
- **Status.** Unlogged until this entry. Disclosed here. Intervals with seed 0 are not reported.

### D26. No AC or battery state in env.json (recorded once at the start of the runs, not by the tool)

- The prereg (Common settings) asks for the power state in each run's `env.json`. `scripts/envinfo.py`
  and `src/passk/envinfo.py` do not record it, and no thermal state either. Neither E3 `env.json`
  has a power key.
- A `pmset` check was made once, when the compute queue started, and recorded AC Power. That
  check is not in the run records, and it was not logged per run.
  Thermal state was not recorded.
- **Status.** The tool still does not record the power state. It may be added later as a new key;
  that would not change existing results.

### D27. E3 ran from a dirty working tree; provenance now records a digest of it

- Both E3 `env.json` files record `git_sha` 096ccb8, `git_dirty` true, and no digest (dirty, no
  digest recorded). The working tree had uncommitted changes. `git_dirty` has been in `env.json` since
  the environment record was added, so the flag was present; what was missing was which dirty
  state a run came from.
- **Change.** `env.json` now also records `git_diff_sha256`: a SHA-256 over the binary diff against
  HEAD and the `git status --porcelain` list. It covers tracked changes and the names of untracked
  files, not their contents. Runs made after this change (E4 and E5) carry it. The E3 files do not,
  because their records were written before the key existed.
- **Effect on results.** None. It is an added key only.

### D28. Strict argument mode is opt-in and untested in experiments

- See D24 for the change and the tests. Strict mode is not used by any pre-registered run. Failure
  counts such as `schema_error` under strict mode would be higher than under the lenient default
  and cannot be compared with the committed E3 runs without a new run.

## Results rendered, E5 reversed gate, timing, unrun arms

### D29. Timing measured on a shared laptop; wall times are not comparable across runs

- The per-episode wall times and token totals in the results block are measured on a laptop that
  ran other jobs. The E3 0.5B run overlapped with benchlint E2 and E3 jobs for part of its run
  (noted during the runs). Its wall times are therefore not comparable with the 1.5B run,
  and no run's wall times are a benchmark. The block states this next to each timing line.
- The power state was not recorded per run (D26). A pmset check at the start of the compute queue
  recorded AC power. Whether the E4 and E5 runs overlapped with other jobs is not recorded.

### D30. Reversed E5 gate: a post-hoc demonstration, not pre-registered

- `results/gates/e5_reversed_v2_baseline_v1_candidate.txt` is the gate applied to the same two E5
  runs with the roles swapped, so the v2 run is the baseline. It reports FAIL, because the drop
  exceeds the policy's limit. The pre-registered comparison is `results/gates/e5_v1_vs_v2.txt`,
  which reports PASS.
- The reversed run was made after the pre-registered comparison, to show the gate firing on a drop.
  It is not in the pre-registration and is not a second E5 result. The renderer labels it "POST-HOC
  demonstration, NOT pre-registered" wherever it appears, and the README says the same. The gate
  was not changed.
- The E5 verdict is reported as measured. The result (the candidate's pass^3 is above the baseline's)
  is surprising, and its cause was not tested (see the README Findings section).

### D31. E6 and the Qwen3-1.7B arm were not run

- E6 (the transient-failure stretch) was not run. It is a stretch goal, and it was not run at
  that point. Its config exists (`configs/e6_flaky.yaml`), and D16 made `flaky_p` settable from
  `RunConfig`, so it is runnable later.
- The Qwen3-1.7B arm was not run. The decode-speed check in D19 cleared the 8 tok/s rule. The
  repository records no reason for not running the arm. Its config (`configs/e3_qwen3_17b.yaml`) is
  unchanged.
- E7 (the hosted arm) was not completed because the free-tier daily quota ran out (D21, D22). No E7
  results are committed.

## D32. Platform-stable number formatting

Found after publication, when CI on Python 3.12 printed `baseline 0.137` for a value that prints
`0.138` on the author's machine. The E5 baseline pass^3 is exactly 0.1375, a rounding tie, and
Python 3.12 sums floats more accurately, so `f"{x:.3f}"` landed on either side of the tie. Text
formatting in the gate reasons, the CLI, the site and the README renderer now goes through
`passk.numfmt.fixed`, which rounds ties half away from zero. No measured value changed. The two
stored gate outputs, `results/headline.json` and the README block were regenerated; the E5
baseline reads 0.138 (was 0.137) and one pass^2 table cell reads 0.538 (was 0.537).

## Qwen3 arm, E6, E7b and the headline

### D32. Qwen3-1.7B arm: timing check, and the smoke evidence

- The Qwen3-1.7B Q8_0 arm (`configs/e3_qwen3_17b.yaml`, `/no_think`, n_ctx 4096) ran at n = 5, 24
  tasks, complete. The timing check (the 40 s median rule, applied to this arm before the full run)
  is recorded in the results block for the Qwen3 run: the median episode wall time is below 40 s, so
  the arm ran at n = 5 as configured.
- The check before the full run used smoke episodes of 13 to 28 s, as recorded during the runs.
  **That smoke output is not in this repository.** `runs/` holds only the 0.5B E3 smoke run and the
  E7 smoke run, so the smoke figures cannot be checked here. The completed run is the evidence in
  this repository.
- The completed run has one slow episode. Its wall time is in the trajectories, and the block's
  median is not affected by it. It was not investigated.
- The 40 s rule was written for the 1.5B arm (D9). Applying it to the Qwen3 arm is a choice made
  here, not a pre-registered rule for that arm.

### D33. E6 ran as pre-registered (1.5B, flaky_p 0.2, n = 3)

- E6 (`configs/e6_flaky.yaml`) ran with 24 tasks x 3 trials, the design in `docs/prereg.md`. It is a
  descriptive robustness run. It is compared with the E3 1.5B baseline (5 trials per task) without a
  significance claim, because the trial counts differ and the runs are not paired.
- The recovery figure in the results block is the pre-registered measure: the share of episodes with
  at least one transient failure that still succeeded. The renderer computes it from the
  trajectories and labels it.

### D34. E7b under amendment 3; the free-tier quota used

- E7b (`configs/e7b_hosted_stratified.yaml`) ran under amendment 3 of `docs/prereg.md`, which was
  committed before the run. It has three tasks (the first of each tier, `t1_01`, `t2_01`, `t3_01`),
  three trials each, nine episodes. It is complete.
- It used 48 model calls, against the account's 50 free requests per day (the count is in the
  results block). Each model call is one step of an episode that makes a model decision.
- The model id reported by the provider equals the requested id. Free-tier routing and model
  versions can change between days, so the id is recorded per run and stated in the block.
- All nine episodes succeeded. This is a small descriptive sample. It supports no general claim.
- The full E7 design (24 tasks x 3 trials) was not run. It needs more free requests per day than the
  account allows (D21, D22). No E7 full-design results are committed.

### D35. Headline items and the formatting of new numbers

- The hub headline has six items, and exactly three are featured: the E3 1.5B item, the Qwen3 item and
  the E5 gate item. The E3 0.5B item is not featured. E6 and E7b are in the README and the site, not in
  the headline, because the item cap is full.
- Every number the renderer prints goes through `passk.numfmt.fixed`, which rounds ties half away
  from zero after removing float noise. CI on Python 3.12 had rounded a tie differently (commit
  `4845ea3`), so raw format strings were replaced.
- The E6 and E7b tables are descriptive. Neither has a significance test, and each says why.
