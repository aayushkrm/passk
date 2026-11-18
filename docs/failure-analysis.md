# Failure analysis

Status: **not written yet.** This file is filled in by hand after the E3 runs exist. It is a
reading of trajectories, not a measurement, and it is kept apart from the generated results.

## Instructions for the author

1. Run E3 first (`configs/e3_qwen05.yaml`, `configs/e3_qwen15.yaml`) and commit the reports
   under `results/e3/`. Do not start this file before those results exist.
2. Choose 10 trajectories with a fixed rule, stated here before reading them. Suggested rule:
   take the first failed trial, in task-id order, for each of the 10 tasks with the most
   failures, across both models. Record the rule and the seed of each trajectory.
3. For each trajectory, write:
   - the task id, tier, model, trial and seed;
   - the primary failure category from `passk.taxonomy`, and whether you agree with it;
   - the step at which the trajectory first went wrong, with the action and the tool result;
   - whether the failure is the model's (wrong reasoning or argument), the harness's (for
     example a schema error caused by the grammar or the prompt), or the environment's;
   - one sentence on what would change the outcome.
4. Count the categories across the 10 and say what that count does and does not support. Ten
   trajectories cannot establish a rate.
5. Do not add numbers that are not in `results/`. Refer to the trajectory by file and line.

## Known harness-level issues to check first

- Schema errors: check whether the tool name or an argument type is wrong, and separate that
  from a wrong choice of tool or argument (see `docs/deviations.md`, D14).
- Argument types, such as an integer where the environment expects a string id.
- Episodes that reach `max_turns` after a run of identical failed calls. Check whether they
  are loops or retries.

## Trajectories read

_None yet._
