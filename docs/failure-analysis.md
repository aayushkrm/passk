# Failure analysis: trajectories read by hand

This file records what individual trajectories from the E3 runs look like. It is commentary,
not a result table. Aggregate numbers live in `results/` and the README block, and are not
repeated here. Every trajectory below is quoted from
`results/e3/<run>/trajectories.jsonl` by run, task and trial, so each can be re-read.

Who read them: the orchestrating AI assistant, not Ayush. Ayush should re-read at least the
trajectories marked **(check)** before relying on the interpretations, because the
interpretations are mine and the rows are data.

Runs referred to:

- `e3-qwen2.5-0.5b-t0.7` (Qwen2.5-0.5B-Instruct Q4_K_M, prompt v1, T=0.7, MCP stdio)
- `e3-qwen2.5-1.5b-t0.7` (Qwen2.5-1.5B-Instruct Q4_K_M, same settings)

Steps are shown as `[index] tool(arguments) -> outcome`. Only the first steps are quoted when an
episode repeats itself.

## How scoring treats things you might not expect

1. **Success is a state check, not "did the agent say it was done".** An episode that never
   emits a `final` action can still succeed if the final state satisfies the goal and no policy
   was violated (0.5B `t1_04` trial 4 and trial 1: one correct `cancel_order`, then the same
   call three more times with `ERR order O2 is already cancelled`; scored success).
2. **A cancelled duplicate still counts as a second order.** `t1_08` forbids two orders for
   `C12`, and the environment keeps cancelled orders in the order list. 1.5B `t1_08` trial 2
   placed one order, placed a second, and cancelled the second. The final state has `O1`
   (placed) and `O2` (cancelled), so the forbidden clause fires and the episode is a failure
   (`max_turns`). This is a deliberate "no extra side effects" rule, but it is strict, and a
   different environment could reasonably treat a cancelled duplicate as harmless.
3. **The failure label is one primary category per episode, by precedence.** It names the first
   matching rule, not the root cause (see 1.5B `t3_02` below, where the label is
   `schema_error` although the agent recovered from that error).

## Qwen2.5-0.5B: what a near-floor model does

**0.5B `t3_04` trial 0 (`tool_error_unrecovered`).** Ten consecutive
`place_order({"compound_id": "200", "grams": 200}) -> ERR unknown compound id '200'`. The model
never calls `search_compounds` to find an id, and repeats the failing call with identical
arguments until the turn limit. It saw the error each time (tool results are in its history).

**0.5B `t1_08` trial 3 (`tool_error_unrecovered`).**
`place_order({"compound_id": "C12 (Sodium azide)", ...}) -> ERR unknown compound id` four
times. The id and the name were fused into one string, and the model did not correct it after
the error.

**0.5B `t1_02` trial 1 (`loop`).** Four `place_order({"compound_id": "C03", "grams": 40})`
calls, each succeeding (`O1` to `O4`). The model never emitted a `final` action, so it kept
placing the same order. Side effects accumulate because each repeat succeeds.

**0.5B `t3_01` trial 3 (`policy_violation`).** Ten identical `place_order` calls on the
controlled compound `C09`, each rejected with `controlled compound C09 has no approved
request_approval`, and each recorded as a violation. In the same task, trial 4 did call
`request_approval` (reason text copied from the error message), but then called `place_order`
again and again until the budget reached 0.00 (`budget_exceeded` violations).

**0.5B `t1_07` trial 0 (`schema_error`).** The model put the tool arguments one level too deep:
`get_compound({"compound_id": "C11", "arguments": {}})` (accepted, extra key ignored), then
`place_order({"compound_id": "C11", "arguments": {"grams": 5.0}})`, which the server rejects
with `grams: Field required`. The same malformed call repeats. **(check)** whether the v1
format description makes the nesting ambiguous: the action JSON has an `arguments` key, and
the model copies that key name into the tool arguments.

Pattern across these: the 0.5B model rarely reacts to an error. It repeats the failing call
instead of changing the id or looking one up.

## Qwen2.5-1.5B: where an agent that sometimes succeeds fails

**1.5B `t2_05` trial 3 (success).** The path the task wants:
`search_compounds("sodium chloride")`, `get_compound(C06)`, `place_order(C06, 45.0)`,
`get_budget()`, `final("The remaining budget is 99.1.")`. Search first, then act, then read
back, then stop.

**1.5B `t2_08` trial 4 (success)** is the same path on a toxic compound
(`search`, `get_compound`, `place_order`, `final`), and **1.5B `t2_08` trial 1
(`false_success_claim`)** is the same task failing: after the same first three steps the model
called `request_approval` for a non-controlled compound, then `cancel_order("O1")` (undoing the
order it had just placed), then emitted `final("Total budget: $45.0, spent: $0.0, remaining:
$45.0")`. The message is a plausible final report, but the state no longer satisfies the goal,
because the order was cancelled. The same model, prompt and task gives a success and a failure.

**1.5B `t3_08` trial 2 (`policy_violation`).** The task involves two controlled compounds
(`C13`, `C09`). After `search_compounds` for each, the model called `place_order` on `C13`,
`C09`, `C13`, `C09`, `C13` and every call was rejected with `controlled compound ... has no
approved request_approval`. It never called `request_approval`, even though the error names the
missing step. The episode ended with eight recorded violations. Contrast 1.5B `t3_02` trial 0,
where `request_approval` came before `place_order`.

**1.5B `t3_04` trial 3 (`policy_violation`).** `place_order(C02, 200 g)` was rejected because it
cost 70.00 with 60.00 remaining. The model then called `request_approval` for a compound that
needs none (status `not_required`), retried the same order, and finally called
`cancel_order({"order_id": "order_id"})`, i.e. it copied the schema field name as the value. The
model did not lower the quantity, which was the one change that would have met the budget.

**1.5B `t2_04` trials 0 and 2 (`tool_error_unrecovered`).** The first action in both is
`cancel_order` with an order id that is really a compound name (`"Toluene"`,
`"toluene_order_id"`). The model then calls `get_compound({"compound_id": "Toluene"})` six times
in a row, each `ERR unknown compound id`, without calling `search_compounds`. This task is 0 of 5
in this run.

**1.5B `t2_06` trial 1 (`max_turns`).** `search`, `get_compound`, `place_order(C11, 10)`,
`get_budget()`, then `cancel_order(O1)` and `place_order(C11, 10)` again. The model undoes and
redoes its own work. The goal is never held at the moment the turns run out.

**1.5B `t1_08` trial 2 (`max_turns`).** See "How scoring treats" item 2: the cancelled
duplicate order is what makes this a failure.

**1.5B `t3_02` trial 3 (`schema_error`).** After the correct `search_compounds`, the model called
`get_compound({"id": "C13"})` (wrong key; rejected), then recovered: `request_approval`,
`place_order(C13, 10)`, both succeeded. Then it called `cancel_order("O1")` on the order it had
just placed. The primary label is `schema_error` because a schema error appears in the episode
(the taxonomy counts it even when recovered), but the reason the goal is unmet is the cancel.
**(check)** whether the precedence rule should require the schema error to be the last unrecovered
error before it takes priority. See `docs/design.md`, "taxonomy rules".

## What these trajectories suggest (interpretations, not measured claims)

- **Stopping and undoing.** Several 1.5B failures are not wrong first moves: the model places
  the right order and then cancels or repeats it (`t2_08` trial 1, `t2_06` trial 1,
  `t3_02` trial 3, `t1_08` trial 2). The harness measures this because success is a final-state
  check. A scorer that only checked "did a correct `place_order` appear" would score these as
  passes.
- **Names versus ids.** Both models pass a human-readable name where an id is required, and
  neither reliably calls `search_compounds` first when the instruction contains only a name.
- **Error text is not enough.** The policy errors name the missing step, and the 1.5B model still
  repeats the rejected call (`t3_08` trial 2). The 0.5B model repeats failing calls verbatim.
- **Not tested here:** whether a longer system prompt, few-shot examples, or a retry budget
  changes these patterns. The v1 prompt is the pre-registered one, and the v1-versus-v2 contrast
  in E5 removes only the policy section.

## Open questions for the author

1. Should a cancelled duplicate order be treated as harmless? Changing it changes pass rates, so
   it would need a new pre-registered task version, not an edit.
2. Should the taxonomy label the last unrecovered error instead of the first matching rule?
3. Is the `arguments` key in the action schema confusing for small models (0.5B `t1_07`)?
