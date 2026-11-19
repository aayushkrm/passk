# Limitations

This file lists what the project does not show. The README links here from its results
section. Read it before drawing a conclusion from any number.

## Scope of the claims

- **One domain.** Every task is a lab-inventory episode in `labstock`. Results say nothing
  about other tool-using domains, other tool sets or real inventory systems.
- **Two small models.** The model results are for Qwen2.5-0.5B-Instruct and
  Qwen2.5-1.5B-Instruct, Q4_K_M quantisation, and any other model that is later added with its
  own config. They do not say anything about "LLMs" in general.
- **Single-instruction episodes.** There is no simulated user and no multi-turn dialogue. A task
  is one instruction, then tool calls, then a final message.
- **No LLM judge.** Success is a deterministic state check. Tasks that need a judgement about
  the quality of a message are out of scope.
- **Synthetic calibration.** E1 and E2 test the estimator and the gate against scripted agents
  whose ground truth is known. They show the tool behaves as its maths says. They do not show
  that the gate catches a regression in a real model.

## Statistics

- **Sample sizes are small.** E3 uses 24 tasks and 5 trials per task. A 95% interval on a
  per-task proportion from 5 trials is wide, and the bootstrap over 24 tasks is only as good as
  that many tasks. The study can detect large changes only. The README states the interval
  for every number.
- **Power of the gate.** At n = 5 and k = 3 the power figure in E2 is a property of this
  design and this policy, not of the gate in general. A smaller change in the candidate would
  be missed more often.
- **No multiplicity correction.** Several metrics and several k values are reported. No single
  one is treated as the headline test, and no claim of a difference is made unless its
  interval excludes 0.
- **Estimator assumptions.** pass^k is estimated from independent trials. Trials of one task
  share a prompt and a model, and at temperature 0 they are identical, so the estimator is not
  meaningful there. E4 checks this directly.

## Hardware and runtime

- **One laptop.** Timings are for an Intel i7-4770HQ laptop (Haswell, 4 cores, 8 threads,
  16 GB), CPU only, `n_threads` = 4. They do not describe other hardware.
- **Wall time depends on load.** The laptop is shared, and the timings in the results are
  medians per episode, not benchmarks.
- **Developed on macOS x86_64.** CI runs on Linux, so the test suite is the only check that
  runs on both.

## Model and grammar

- **Structured output is a constraint, not a guarantee.** The JSON schema is flat with four
  required keys (`kind`, `tool`, `arguments`, `message`). The parser still checks the values
  (a tool name outside the enum, a wrong type) and counts a bad value as a schema error. The
  share of schema errors depends on the schema, so the schema is part of the result. Before
  amendment 1 of the pre-registration, only `kind` was required, and the rate was much higher
  (`docs/deviations.md`, D14).
- **Quantisation.** Q4_K_M is one quantisation. Other quantisations and the fp16 model may
  behave differently.
- **Model drift.** A hosted model (E7) can change without notice, and the free tier can be
  rate limited.
- **Hosted quota.** The free tier allows 50 model requests per day per account. The full E7 run
  (24 tasks x 3 trials) was not completed on that quota (`docs/deviations.md`, D21). E7 is
  reported as partial if it does not finish. Its results carry the date and the provider-reported model id, and they are
  descriptive only.

## Missing pieces

- **E6 (recovery from transient failures)** is a descriptive run with 3 trials per task. It is
  compared with the 5-trial E3 baseline without a significance claim (`docs/deviations.md`, D33).
- **Reading of failures.** `docs/failure-analysis.md` is a hand reading of a sample of
  trajectories. The sample is small, and it is a reading, not a measurement.
- **Tool-description lint.** The check for poisoned or drifted MCP tool descriptions is not
  implemented. Nothing in this repository checks tool descriptions.

## Software

- **MCP v2 SDK.** The transport follows the 2026-07-28 specification as implemented in the
  `mcp` 2.x package. A future SDK change could alter the handshake or the metadata used to
  recover tool parameters.
- **No OpenTelemetry SDK.** Traces are JSONL with OpenTelemetry GenAI attribute names. They are
  not exported to a collector.
- **Not on PyPI.** The package is installed from the GitHub repository or from a release wheel.

## Undeclared argument names (strict mode)

- The default is lenient on both transports: an argument name the tool does not declare is dropped.
  The E3 runs used this behaviour over MCP, so the committed results depend on it. The
  `strict_args` option rejects such names, but no experiment uses it. Under strict mode, failure
  counts such as `schema_error` would be higher, so the committed failure counts may be lower than
  they would be under strict mode. Treat them as a lenient-mode result (`docs/deviations.md`, D24).
