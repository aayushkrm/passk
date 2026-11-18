"""Frozen interfaces for passk.

This module is FROZEN (DECISION.md §4.3). Only the orchestrator may change it.
Builders needing a change must request it in their report and keep a local adapter.

All data types are pydantic v2 models that round-trip through JSON. Every top-level
report carries ``schema_version``. The ``Env``, ``Agent`` and ``ToolTransport``
protocols describe the structural interfaces used by the runner.
"""

from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """Description of one tool an environment exposes to the agent.

    Attributes:
        name: Tool name, unique within an environment.
        description: Natural-language description shown to the agent.
        parameters: JSON Schema (object) for the tool arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A single tool invocation requested by the agent.

    Attributes:
        call_id: Identifier unique within a trajectory.
        name: Name of the tool to call.
        arguments: Arguments object passed to the tool.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Outcome of a ``ToolCall``.

    Attributes:
        call_id: Identifier of the call this result answers.
        ok: Whether the call succeeded.
        content: Tool output (any JSON-serialisable value).
        error: Error message when ``ok`` is False, otherwise None.
    """

    call_id: str
    ok: bool
    content: Any
    error: str | None = None


class AgentAction(BaseModel):
    """One decision made by an agent at a turn.

    Attributes:
        kind: ``"tool_call"`` to invoke a tool, ``"final"`` to end the episode.
        tool_call: The call to perform when ``kind == "tool_call"``.
        message: Final message to the user when ``kind == "final"``.
    """

    kind: Literal["tool_call", "final"]
    tool_call: ToolCall | None = None
    message: str | None = None


class Step(BaseModel):
    """One turn of an episode: the agent's action and the environment's result.

    Attributes:
        index: Zero-based turn index.
        action: The action the agent took.
        result: Tool result for a ``tool_call`` action, None for ``final``.
        latency_s: Wall time spent producing the action, in seconds.
        input_tokens: Prompt tokens for the model call, if the agent is a model.
        output_tokens: Generated tokens for the model call, if the agent is a model.
    """

    index: int
    action: AgentAction
    result: ToolResult | None = None
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Assertion(BaseModel):
    """A check on the final environment state, evaluated by ``goals.py``.

    Attributes:
        path: Dotted path with optional filters, e.g. ``orders[compound_id=C07].grams``.
        op: Comparison operator.
        value: Expected value (unused by ``absent``).
    """

    path: str
    op: Literal["equals", "contains", "not_contains", "count_eq", "absent"]
    value: Any = None


class Task(BaseModel):
    """A single evaluation task.

    Attributes:
        id: Stable task identifier.
        tier: Difficulty tier (1 = single call, 2 = lookup then act, 3 = policy-constrained).
        instruction: The user instruction given to the agent.
        initial_state: Environment state at reset.
        goal: Assertions that must hold for success.
        forbidden: Assertions that must not hold (violations if they do).
        reference_solution: Known-correct sequence of tool calls.
        max_turns: Turn budget for the episode.
        tags: Free-form labels.
    """

    id: str
    tier: Literal[1, 2, 3]
    instruction: str
    initial_state: dict[str, Any]
    goal: list[Assertion]
    forbidden: list[Assertion]
    reference_solution: list[ToolCall]
    max_turns: int = 8
    tags: list[str] = Field(default_factory=list)


class FailureCategory(StrEnum):
    """Primary failure label for a trajectory.

    Members are listed in precedence order. The taxonomy assigns the first category
    whose condition holds: ``policy_violation`` > ``schema_error`` >
    ``tool_error_unrecovered`` > ``false_success_claim`` > ``loop`` > ``max_turns`` >
    ``wrong_state`` > ``success``.
    """

    POLICY_VIOLATION = "policy_violation"
    SCHEMA_ERROR = "schema_error"
    TOOL_ERROR_UNRECOVERED = "tool_error_unrecovered"
    FALSE_SUCCESS_CLAIM = "false_success_claim"
    LOOP = "loop"
    MAX_TURNS = "max_turns"
    WRONG_STATE = "wrong_state"
    SUCCESS = "success"


class Trajectory(BaseModel):
    """The full record of one episode.

    Attributes:
        run_id: Run this episode belongs to.
        task_id: Task that was attempted.
        trial: Trial index for this task (0-based).
        seed: Seed used for the episode, derived from the base seed, task and trial.
        agent_id: Identifier of the agent that acted.
        steps: Ordered turns of the episode.
        final_state: Environment state after the episode ended.
        success: Whether all goal assertions held and no forbidden assertion held.
        violations: Human-readable policy violation messages.
        failure: Primary failure category (``success`` when ``success`` is True).
        wall_s: Total wall time of the episode, in seconds.
    """

    run_id: str
    task_id: str
    trial: int
    seed: int
    agent_id: str
    steps: list[Step]
    final_state: dict[str, Any]
    success: bool
    violations: list[str]
    failure: FailureCategory
    wall_s: float


class AgentConfig(BaseModel):
    """Which agent to run and its parameters.

    Attributes:
        kind: Agent implementation to construct.
        params: Constructor parameters for that implementation.
    """

    kind: Literal["oracle", "noisy_oracle", "llamacpp", "replay"]
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Configuration for one ``passk run``.

    Attributes:
        run_id: Name of the run; also the output directory name under ``runs/``.
        env: Environment name (only ``labstock`` in v0.1).
        tasks_glob: Glob of task YAML files.
        agent: Agent configuration.
        n_trials: Trials per task.
        base_seed: Base seed for deriving episode seeds.
        transport: How the agent reaches the environment tools.
        k_values: Values of k for pass@k and pass^k.
    """

    run_id: str
    env: str = "labstock"
    tasks_glob: str
    agent: AgentConfig
    n_trials: int
    base_seed: int
    transport: Literal["inprocess", "mcp_stdio"]
    k_values: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])


class TaskStats(BaseModel):
    """Aggregated results for one task across its trials.

    Attributes:
        task_id: Task identifier.
        tier: Task tier.
        n: Number of trials.
        c: Number of successful trials.
        pass_at_k: Map from k to the unbiased pass@k estimate.
        pass_hat_k: Map from k to the unbiased pass^k estimate.
        failures: Count of each failure category among the trials.
    """

    task_id: str
    tier: Literal[1, 2, 3]
    n: int
    c: int
    pass_at_k: dict[int, float]
    pass_hat_k: dict[int, float]
    failures: dict[str, int]


class RunReport(BaseModel):
    """Top-level report written to ``runs/<run_id>/report.json``.

    Attributes:
        schema_version: Version of this report schema. Always ``"1"`` in v0.1.
        run_id: Run identifier.
        config: The configuration that produced the report.
        env_info: Environment record (see ``scripts/envinfo.py``).
        tasks: Per-task statistics.
        summary: Metric name -> k -> ``{"mean", "ci_low", "ci_high"}``.
        failure_totals: Failure category name -> count across all trials.
        tokens: Token usage totals and medians.
        wall_s: Total wall time of the run, in seconds.
    """

    schema_version: Literal["1"] = "1"
    run_id: str
    config: RunConfig
    env_info: dict[str, Any]
    tasks: list[TaskStats]
    summary: dict[str, dict[int, dict[str, float]]]
    failure_totals: dict[str, int]
    tokens: dict[str, Any]
    wall_s: float


class GatePolicy(BaseModel):
    """Policy for ``passk gate``.

    Attributes:
        k: The k at which pass^k is compared.
        max_drop: Largest tolerated drop in pass^k (candidate vs baseline).
        min_pass_hat_k: Absolute floor on candidate pass^k, or None.
        require_significance: If True, a drop also needs a paired-bootstrap CI excluding 0.
        alpha: Significance level; the CI is ``1 - alpha`` wide.
    """

    k: int
    max_drop: float
    min_pass_hat_k: float | None = None
    require_significance: bool = True
    alpha: float = 0.05


class GateDecision(BaseModel):
    """Result of ``passk gate``.

    Attributes:
        passed: Whether the candidate passes the policy.
        reasons: Human-readable reasons for the decision.
        delta: Paired difference in pass^k (candidate minus baseline).
        ci_low: Lower bound of the paired-bootstrap CI for ``delta``.
        ci_high: Upper bound of the paired-bootstrap CI for ``delta``.
    """

    passed: bool
    reasons: list[str]
    delta: float
    ci_low: float
    ci_high: float


class Env(Protocol):
    """Structural interface of an environment.

    ``reset`` mints an ``env_handle`` that identifies one episode's state. Every later
    call takes that handle, so the environment holds no hidden global state.
    """

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode for ``task`` and return its server-minted handle."""
        ...

    def tools(self) -> list[ToolSpec]:
        """Return the tools this environment exposes."""
        ...

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute ``call`` against the episode identified by ``handle``."""
        ...

    def state(self, handle: str) -> dict[str, Any]:
        """Return the current state of the episode identified by ``handle``."""
        ...

    def violations(self, handle: str) -> list[str]:
        """Return policy violations recorded so far in the episode."""
        ...


class Agent(Protocol):
    """Structural interface of an agent."""

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Choose the next action given the task, history so far and available tools."""
        ...


class ToolTransport(Protocol):
    """How tool calls reach an environment (in-process or over MCP stdio).

    Mirrors the ``Env`` protocol method for method.
    """

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode for ``task`` and return its handle."""
        ...

    def tools(self) -> list[ToolSpec]:
        """Return the tools available over this transport."""
        ...

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute ``call`` for the episode identified by ``handle``."""
        ...

    def state(self, handle: str) -> dict[str, Any]:
        """Return the episode state identified by ``handle``."""
        ...

    def violations(self, handle: str) -> list[str]:
        """Return policy violations recorded for the episode identified by ``handle``."""
        ...
