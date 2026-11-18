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
  draws, and the runner calls it before each episode (already in the scaffold; now covered by
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
- **Decision:** not changed. The schema is part of the pre-registered agent, and `schema_error`
  is a measured category, so changing it now would be a prompt or schema change after the
  pre-registration. The orchestrator must decide before E3. If `tool` is made required,
  that is a logged amendment to `docs/prereg.md`, and the smoke runs are repeated under the new
  schema.

## At and after the pre-registration commit

### D6. E6 cannot run

- `flaky_p` is set on `LabStockEnv`, but `RunConfig` has no field for it, and the schema is
  frozen. E6 stays in the pre-registration as a stretch and is not run. A schema change, approved
  by the orchestrator, is needed first.

### D7. Qwen3-1.7B arm not added

- The optional third arm needs a measured decode speed of at least 8 tok/s. No spike record
  exists in the repository (`spike/spike.json` is absent), and no measurement was made here. The
  arm is not added. The decision is recorded here and not in a config.

### D8. Qwen2.5-1.5B weights are not on disk

- `~/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/` holds only a download cache. The E3 1.5B config,
  the E5 config and the 1.5B arm of the timing rule cannot run until the Q4_K_M file is present.
  The configs assume `~/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf`.

### D9. Timing rule for the 1.5B arm (not yet decided)

- The 40 s and 90 s rules in `docs/prereg.md` depend on the median 1.5B episode time, which
  cannot be measured until D8 is resolved. From the 0.5B smoke runs, each model call takes a few
  seconds, and an episode makes several calls, so the 1.5B median is likely above 40 s. The
  n = 3 change, if it is needed, is logged when the rule is applied.

### D10. Model SHA-256 and repo id in env.json (added)

- `LlamaCppAgent` records `repo_id`, the file name and the SHA-256 of the model file in the
  run metadata, so `env.json` carries them (DECISION.md section 6.2). Runs made before this
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
