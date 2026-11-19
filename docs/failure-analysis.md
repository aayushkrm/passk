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
(`C13`, `C09`). After `search_compounds` for `CS-77` and `CS-41`, the model made eight
`place_order` calls, alternating `C13`, `C09`, `C13`, `C09`, `C13`, `C09`, `C13`, `C09`. Each was
rejected with `controlled compound ... has no approved request_approval`. It never called
`request_approval`, even though the error names the missing step. The episode has eight recorded
violations, one per rejected order. Contrast 1.5B `t3_02` trial 0, where `request_approval` came
before `place_order`.

**1.5B `t3_04` trial 3 (`policy_violation`).** `place_order(C02, 200 g)` was rejected because it
cost 70.00 with 60.00 remaining. The model then called `request_approval` for a compound that
needs none (status `not_required`), retried the same order, and finally called
`cancel_order({"order_id": "order_id"})`, i.e. it copied the schema field name as the value. The
model did not lower the quantity, which was the one change that would have met the budget.

**1.5B `t2_04` trials 0 and 2 (`tool_error_unrecovered`).** The first action in both is
`cancel_order` with an order id that is really a compound name (`"Toluene"` in trial 0,
`"toluene_order_id"` in trial 2). In trial 0 the model then calls
`get_compound({"compound_id": "Toluene"})` seven times in a row (steps 1 to 7), each
`ERR unknown compound id`, without calling `search_compounds`. In trial 2 the second action is
`request_approval` for `"toluene"`, which is rejected the same way, and then
`get_compound({"compound_id": "toluene"})` is called six times in a row (steps 2 to 7). Both episodes stop at eight steps, the turn budget of this task, and are labelled
`tool_error_unrecovered`, which ranks above `max_turns` in the taxonomy. This task is 0 of 5 in
this run.

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

## E5: prompt v2 on the 1.5B model (four trajectories read by hand)

Four v2 trajectories from `results/e5/e5-qwen2.5-1.5b-v2-t0.7/trajectories.jsonl` were read, and the
counts below were checked by script. The v1 counts come from `results/e3/e3-qwen2.5-1.5b-t0.7`.

- **`t2_06` trial 0 (success; v1 0 of 5, v2 5 of 5).** Five steps, no errors: `search_compounds` for
  `citric acid`, `get_compound` for `C11`, `place_order` for 10 g, `get_budget`, then the final
  message "Your order for 10 g of Citric acid has been placed. You have $100.0 remaining in your
  budget."
- **`t2_03` trial 0 (success; v1 1 of 5, v2 5 of 5).** Four steps, no errors, ending with the final
  message "Total budget: 100.0, Spent: 2.0, Remaining: 98.0".
- **`t3_01` trial 0 (`policy_violation`).** The first `place_order` for `C09` was rejected with
  "policy violation: controlled compound C09 has no approved request_approval". The model then called
  `request_approval` with the reason "QC run request", and its second order succeeded. The episode is
  still labelled `policy_violation`, because the violation stays on the record
  (`order_without_approval: C09 10 g`).
- **`t3_08` trial 0 (`tool_error_unrecovered`).** Ten steps, nine of them errors. The model passed
  descriptive text as ids (for example "controlled standard CS-77" as a compound id and "not_required"
  as an order id). The label is `tool_error_unrecovered`, which ranks above `max_turns`.

These four episodes show that v2 still produces policy violations: the `t3_01` episode orders a
controlled compound before approval. They do not show why v2 succeeds more often, and four
episodes cannot establish a mechanism.

## Qwen3-1.7B arm (Q8_0, `/no_think`): four trajectories read by hand

Read from `results/e3/e3-qwen3-1.7b-q8-t0.7/trajectories.jsonl`. The step counts and error counts
were checked by script. The 1.5B comparisons are the same task and trial in
`results/e3/e3-qwen2.5-1.5b-t0.7`.

- **`t1_01` trial 0 (success).** Three steps, no errors: `request_approval` for `C07` with the reason
  "Ordering 25g of Glycerol for lab use.", `place_order` for `C07` at 25 g, then the final message
  "Order placed successfully. Order ID: O1. Cost: $1.25."
- **`t3_01` trial 0 (success).** Three steps, no errors: `request_approval` for `C09` with the reason
  "QC run", `place_order` for `C09` at 10 g, then the final message "Order O1 placed successfully.
  Budget: total 100, spent 25, remaining 75."
- **`t1_04` trial 0 (`false_success_claim`).** One step, and it is a final message with no tool call:
  "Order O2 canceled." The 1.5B run on the same task and trial made four tool calls and ended with
  `tool_error_unrecovered`.
- **`t3_02` trial 0 (`tool_error_unrecovered`).** Two steps. `request_approval` for `CS-77` returned
  "unknown compound id 'CS-77'", and the next step was the final message "Compound ID 'CS-77' is not
  recognized. Please provide a valid compound ID." There was no retry. The 1.5B run on the same task
  and trial made eight steps and ended with `schema_error`.

In these four episodes the Qwen3 runs are short: one to three steps. On the two tasks where the 1.5B
run was compared, it took four steps (`t1_04`) and eight steps (`t3_02`). The two failures end the episode without a recovery attempt. Only these
four were read, so nothing here explains the difference in results, and the comparison is confounded
(model family, quantization and thinking mode all differ; see the README).

## E6 (1.5B, injected transient failures at 0.2): two trajectories read by hand

Read from `results/e6/e6-qwen2.5-1.5b-flaky0.2/trajectories.jsonl`. A transient failure is the
environment's `transient` error, injected for a call. The recovery figure is in the results block.

- **`t1_04` trial 1 (`tool_error_unrecovered`).** Step 0, `cancel_order` for `O2`, returned `transient`.
  The model did not repeat that call. Step 1, `get_budget`, also returned `transient`. Step 2 repeated
  the identical `get_budget` call, which succeeded. Step 3 is the final message "Remaining budget:
  $97.55, spent: $2.45, total: $100.00". The episode is labelled `tool_error_unrecovered`, because the
  `cancel_order` failure was never followed by a successful `cancel_order`.
- **`t1_01` trial 1 (`tool_error_unrecovered`).** Step 0 `place_order` succeeded. Step 1,
  `request_approval`, returned `transient` and was not repeated. Step 2, `cancel_order` for `O1`,
  succeeded. Step 3 repeats that `cancel_order` for `O1` and returns `transient`. The episode ends at
  four steps with no final message.

The first episode retries the `get_budget` that failed at step 1 (step 2), but not the `cancel_order`
that failed at step 0. The second episode does not retry the `request_approval` that failed at step 1.
It also shows that a transient failure can strike a repeat of a call that had just succeeded (step 3
repeats the `cancel_order` of step 2). Two episodes cannot show how often either happens.

## E7b (hosted free model, three tasks): one trajectory read by hand

Read from `results/e7b/e7b-nemotron-3-ultra-stratified-t0.7/trajectories.jsonl`.

- **`t3_01` trial 0 (success).** Seven steps and no errors: `search_compounds` for `C09`,
  `search_compounds` for `CS-41`, `get_compound` for `C09`, `request_approval` for `C09` with the reason
  "QC run", `get_budget`, `place_order` for `C09` at 10 g, and the final message "Order placed
  successfully. Ordered 10 g of Controlled standard CS-41 (C09) for QC run. Order ID: O1, Cost: $25.00.
  Remaining budget: $75.00." On this task the Qwen3 run took three steps, and it also made no errors.
  Nine episodes in total are too few for any general statement.
