"""Run tasks x trials for one configuration and write the run directory.

Output layout: ``runs/<run_id>/{trajectories.jsonl, spans.jsonl, report.json, env.json}``.
"""

from pathlib import Path

from passk.schema import Agent, AgentConfig, RunConfig, RunReport


def episode_seed(base_seed: int, task_id: str, trial: int) -> int:
    """Derive the seed for one episode as a stable hash of (base_seed, task_id, trial).

    Args:
        base_seed: Base seed from the run config.
        task_id: Task identifier.
        trial: Trial index.

    Returns:
        A non-negative integer seed. Must not depend on Python's randomised ``hash``.
    """
    raise NotImplementedError("episode_seed: scheduled for work package P-D")


def build_agent(config: AgentConfig) -> Agent:
    """Construct the agent named by ``config``.

    Args:
        config: Agent kind and parameters.

    Returns:
        An object satisfying the ``Agent`` protocol.
    """
    raise NotImplementedError("build_agent: scheduled for work package P-D")


def run(config: RunConfig, out_root: Path = Path("runs")) -> RunReport:
    """Run every task matching ``config.tasks_glob`` for ``config.n_trials`` trials.

    Enforces ``Task.max_turns``, writes the run directory and returns the report.

    Args:
        config: Run configuration.
        out_root: Directory under which ``<run_id>/`` is created.

    Returns:
        The aggregated ``RunReport`` (also written to ``report.json``).
    """
    raise NotImplementedError("run: scheduled for work package P-D")
